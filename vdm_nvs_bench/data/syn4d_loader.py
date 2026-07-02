"""Minimal, dependency-light Syn4D held-out loader (PIL + numpy only).

Ported from recammaster-official's ``Syn4DEvalLoader`` but stripped of the
diffsynth/torch pipeline imports so the workshop repo stays self-contained. Reads
a Syn4D scene's PNG frames + per-frame camera CSV and produces, for a
(source_view, target_view) pair over a common chunk: source frames, target (GT)
frames, and the target camera trajectory (c2w relative to source frame 0).

On-disk layout (per scene):
  <root>/caption/<scene>_per_video_chunks.json
  <root>/<scene>/png/<seq_root>_<view>/<seq_root>_<view>_<frame>.png
  <root>/<scene>/ground_truth/meta_exr_csv/<seq_root>_<view>_camera.csv
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import List

import numpy as np
from PIL import Image

_PNG_FRAME_RE = re.compile(r"^(?P<seq>seq_\d+_\d+)_(?P<frame>\d+)\.png$")


def euler_to_rotation_matrix(yaw, pitch, roll) -> np.ndarray:
    yaw, pitch, roll = np.radians(float(yaw)), np.radians(float(pitch)), np.radians(float(roll))
    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], np.float32)
    r_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]], np.float32)
    r_roll = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]], np.float32)
    zxy_xyz = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], np.float32)
    return (r_yaw @ r_pitch @ r_roll @ zxy_xyz).astype(np.float32)


class Camera:
    def __init__(self, c2w):
        self.c2w_mat = np.array(c2w, dtype=np.float32).reshape(4, 4)
        self.w2c_mat = np.linalg.inv(self.c2w_mat)


def get_relative_pose(cam_params: List[Camera]) -> np.ndarray:
    """[identity, rel...] — each pose expressed relative to cam_params[0]."""
    abs_w2cs = [c.w2c_mat for c in cam_params]
    abs_c2ws = [c.c2w_mat for c in cam_params]
    target0 = np.eye(4, dtype=np.float32)
    abs2rel = target0 @ abs_w2cs[0]
    return np.array([target0] + [abs2rel @ c for c in abs_c2ws[1:]], dtype=np.float32)


class Syn4DEvalLoader:
    def __init__(self, dataset_root, scene_name, seq_root, num_frames, height, width,
                 frame_offset_in_chunk=0, caption_chunk_size=81):
        self.dataset_root = dataset_root
        self.scene_name = scene_name
        self.seq_root = seq_root
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.frame_offset_in_chunk = int(frame_offset_in_chunk)
        self.caption_chunk_size = int(caption_chunk_size)
        self.caption_path = os.path.join(dataset_root, "caption", f"{scene_name}_per_video_chunks.json")
        self.scene_dir = os.path.join(dataset_root, scene_name)
        self.png_root = os.path.join(self.scene_dir, "png")
        self.camera_csv_root = os.path.join(self.scene_dir, "ground_truth", "meta_exr_csv")
        for p in [self.scene_dir, self.png_root, self.camera_csv_root]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing Syn4D path: {p}")
        self._camera_csv_cache: dict = {}
        self._caption_cache = None

    # -- caption chunks -----------------------------------------------------
    def _load_caption_cache(self) -> dict:
        if self._caption_cache is None:
            if os.path.isfile(self.caption_path):
                data = json.load(open(self.caption_path, encoding="utf-8"))
                self._caption_cache = data if isinstance(data, dict) else {}
            else:
                self._caption_cache = {}
        return self._caption_cache

    def _caption_chunks_for_view(self, view_idx) -> dict:
        data = self._load_caption_cache()
        chunks = data.get(f"mp4/{self.seq_root}_{int(view_idx)}.mp4", {})
        return {str(k): str(v) for k, v in chunks.items()} if isinstance(chunks, dict) else {}

    @staticmethod
    def _parse_chunk_tag(tag):
        s, e = str(tag).split("-")
        return int(s), int(e)

    # -- camera csv ---------------------------------------------------------
    def _load_camera_csv(self, view_idx) -> dict:
        if view_idx in self._camera_csv_cache:
            return self._camera_csv_cache[view_idx]
        csv_path = os.path.join(self.camera_csv_root, f"{self.seq_root}_{int(view_idx)}_camera.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing camera csv: {csv_path}")
        rows = {}
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = str(row["name"]).strip()
                if name:
                    rows[name] = {k: float(row[k]) for k in ("x", "y", "z", "yaw", "pitch", "roll")}
        self._camera_csv_cache[view_idx] = rows
        return rows

    # -- png frames ---------------------------------------------------------
    def _seq_png_dir(self, view_idx):
        return os.path.join(self.png_root, f"{self.seq_root}_{int(view_idx)}")

    def _scan_png_frames(self, view_idx):
        seq_dir = self._seq_png_dir(view_idx)
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(f"Missing png dir: {seq_dir}")
        frame_map, frame_width = {}, None
        for name in os.listdir(seq_dir):
            m = _PNG_FRAME_RE.match(name)
            if m is None:
                continue
            frame_map[int(m.group("frame"))] = os.path.join(seq_dir, name)
            if frame_width is None:
                frame_width = len(m.group("frame"))
        return frame_map, (4 if frame_width is None else int(frame_width))

    def choose_chunk_and_frames(self, source_view, target_views, chunk_tag=None):
        src_map, _ = self._scan_png_frames(source_view)
        if not src_map:
            raise ValueError(f"No png frames for source view {source_view}")
        min_num_frames = len(src_map)
        for tv in target_views:
            min_num_frames = min(min_num_frames, len(self._scan_png_frames(tv)[0]))
        if min_num_frames < self.caption_chunk_size:
            raise ValueError(f"{self.scene_name}/{self.seq_root}: only {min_num_frames} frames "
                             f"< caption_chunk_size={self.caption_chunk_size}")
        if chunk_tag is None:
            tag_sets = []
            for v in [source_view] + list(target_views):
                valid = set()
                for tag in self._caption_chunks_for_view(v):
                    try:
                        s, e = self._parse_chunk_tag(tag)
                    except Exception:
                        continue
                    if e < min_num_frames and (e - s + 1) >= self.caption_chunk_size:
                        valid.add(tag)
                if valid:
                    tag_sets.append(valid)
            if tag_sets:
                common = sorted(set.intersection(*tag_sets), key=self._parse_chunk_tag)
                if not common:
                    raise ValueError("No common caption chunks across source/target views.")
                chunk_tag = common[0]
            else:
                chunk_tag = f"0-{self.caption_chunk_size - 1}"
        chunk_start, chunk_end = self._parse_chunk_tag(chunk_tag)
        if chunk_end >= min_num_frames:
            raise ValueError(f"chunk_tag {chunk_tag} exceeds min_num_frames={min_num_frames}")
        max_offset = (chunk_end - chunk_start + 1) - self.num_frames
        if max_offset < 0:
            raise ValueError(f"chunk_tag {chunk_tag} too short for num_frames={self.num_frames}")
        offset = min(max(self.frame_offset_in_chunk, 0), max_offset)
        start = chunk_start + offset
        return chunk_tag, [start + i for i in range(self.num_frames)]

    def _frame_name(self, view_idx, frame_id, width):
        return f"{self.seq_root}_{int(view_idx)}_{int(frame_id):0{int(width)}d}.png"

    def _crop_and_resize(self, img: Image.Image) -> np.ndarray:
        w, h = img.size
        scale = max(self.width / w, self.height / h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        top = max(0, int(round((arr.shape[0] - self.height) / 2.0)))
        left = max(0, int(round((arr.shape[1] - self.width) / 2.0)))
        return arr[top:top + self.height, left:left + self.width]

    def load_video_frames(self, view_idx, frame_ids) -> List[np.ndarray]:
        """List of uint8 (H,W,3) frames, cropped/resized to (height,width)."""
        frame_map, _ = self._scan_png_frames(view_idx)
        frames = []
        for fid in frame_ids:
            if fid not in frame_map:
                raise FileNotFoundError(f"Missing frame {fid} for view {view_idx}")
            frames.append(self._crop_and_resize(Image.open(frame_map[fid]).convert("RGB")))
        return frames

    def load_pose(self, view_idx, frame_id) -> np.ndarray:
        rows = self._load_camera_csv(view_idx)
        _, width = self._scan_png_frames(view_idx)
        name = self._frame_name(view_idx, frame_id, width)
        if name not in rows:
            raise KeyError(f"Missing camera row `{name}` in view {view_idx}")
        r = rows[name]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = euler_to_rotation_matrix(r["yaw"], r["pitch"], r["roll"])
        c2w[:3, 3] = np.array([r["x"], r["y"], r["z"]], dtype=np.float32) / 100.0
        return c2w
