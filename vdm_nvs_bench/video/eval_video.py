"""Video-quality metrics: FVD + CLIP (CLIP-T / CLIP-F / CLIP-V), track-aware.

Reuses the vendored styleganv-I3D FVD (`common_metrics/calculate_fvd.py`) and pip
`open_clip_torch` for CLIP embeddings. Two tracks:

  DAVIS  (no target-view GT): FVD(pred vs source) + CLIP-F + CLIP-T(prompt).
  Syn4D  (paired target GT):  FVD(pred vs gt)     + CLIP-V(source,pred) + CLIP-F + CLIP-T.

FVD is a distribution distance over the whole set of pairs (styleganv I3D features +
Frechet distance); each video must have >= 10 frames.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_VIDEO_DIR = Path(__file__).resolve().parent
_COMMON_METRICS = _VIDEO_DIR / "common_metrics"
DEFAULT_CLIP_MODEL = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
FVD_MIN_FRAMES = 10


def _ensure_common_metrics_import() -> None:
    if str(_COMMON_METRICS) not in sys.path:
        sys.path.insert(0, str(_COMMON_METRICS))


# --------------------------------------------------------------------------
# Video IO
# --------------------------------------------------------------------------
def load_video(path: Path, size: Optional[tuple] = None, max_frames: Optional[int] = None) -> torch.Tensor:
    """Read an mp4 -> (T, C, H, W) float tensor in [0,1]. Optional resize to (H,W)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    frames = []
    for i, frame in enumerate(reader):
        if max_frames is not None and i >= max_frames:
            break
        frames.append(np.asarray(frame)[..., :3])
    reader.close()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,C)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T,C,H,W)
    if size is not None:
        t = torch.nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t


# --------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------
class OpenClipEmbedder:
    def __init__(self, device: torch.device, model_name: str = DEFAULT_CLIP_MODEL) -> None:
        import open_clip
        from torchvision import transforms

        self.device = device
        self.to_pil = transforms.ToPILImage()
        pretrained = None if model_name.startswith("hf-hub:") else "openai"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, precision="fp32", device=device, jit=False
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._text_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text_cache:
            tok = self.tokenizer([prompt]).to(self.device)
            self._text_cache[prompt] = self.model.encode_text(tok, normalize=True).squeeze(0).cpu()
        return self._text_cache[prompt]

    @torch.no_grad()
    def encode_images(self, frames_tchw: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        feats = []
        for start in range(0, frames_tchw.shape[0], batch_size):
            batch = torch.stack(
                [self.preprocess(self.to_pil(f.cpu())) for f in frames_tchw[start:start + batch_size]],
                dim=0,
            ).to(self.device)
            feats.append(self.model.encode_image(batch, normalize=True).cpu())
        return torch.cat(feats, dim=0)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum(dim=-1).mean())


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
def run_video_eval(
    samples: List[dict],
    track: str,
    device: Optional[torch.device] = None,
    clip_model: str = DEFAULT_CLIP_MODEL,
    fvd_size: int = 256,
    fvd_method: str = "styleganv",
    compute_clip: bool = True,
    compute_fvd: bool = True,
) -> dict:
    """Score a list of samples. Each sample: {seq, traj, pred: Path, gt?: Path,
    source?: Path, prompt?: str}. ``track`` in {"davis","syn4d"} selects FVD reference
    (source for davis, gt for syn4d) and whether CLIP-V is computed."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paired_key = "gt" if track == "syn4d" else "source"

    clip_t_vals, clip_f_vals, clip_v_vals = [], [], []
    fvd_pred, fvd_ref = [], []
    embedder = OpenClipEmbedder(device, clip_model) if compute_clip else None

    for s in samples:
        pred = load_video(Path(s["pred"]))
        if compute_clip:
            pf = embedder.encode_images(pred)
            if pf.shape[0] >= 2:  # CLIP-F: adjacent-frame consistency
                clip_f_vals.append(_cos(pf[:-1], pf[1:]))
            if s.get("prompt"):    # CLIP-T: frames vs prompt
                tf = embedder.encode_text(s["prompt"])
                clip_t_vals.append(_cos(pf, tf.unsqueeze(0)))
            if track == "syn4d" and s.get("source"):  # CLIP-V: source vs pred at matched t
                src = load_video(Path(s["source"]))
                sf = embedder.encode_images(src)
                n = min(sf.shape[0], pf.shape[0])
                clip_v_vals.append(_cos(sf[:n], pf[:n]))

        if compute_fvd and s.get(paired_key):
            ref = load_video(Path(s[paired_key]))
            n = min(pred.shape[0], ref.shape[0])
            if n >= FVD_MIN_FRAMES:
                sz = (fvd_size, fvd_size)
                p = torch.nn.functional.interpolate(pred[:n], size=sz, mode="bilinear", align_corners=False)
                r = torch.nn.functional.interpolate(ref[:n], size=sz, mode="bilinear", align_corners=False)
                fvd_pred.append(p)
                fvd_ref.append(r)

    out: dict = {"track": track, "num_samples": len(samples), "clip_model": clip_model}

    def _summ(vals):
        return {"value": float(np.mean(vals)), "std": float(np.std(vals)), "count": len(vals)} if vals else None

    out["clip_t"] = _summ(clip_t_vals)
    out["clip_f"] = _summ(clip_f_vals)
    out["clip_v"] = _summ(clip_v_vals)

    if compute_fvd and len(fvd_pred) >= 2:
        _ensure_common_metrics_import()
        from calculate_fvd import calculate_fvd

        T = min(v.shape[0] for v in fvd_pred)
        v1 = torch.stack([v[:T] for v in fvd_pred], dim=0)  # (N,T,C,H,W)
        v2 = torch.stack([v[:T] for v in fvd_ref], dim=0)
        res = calculate_fvd(v1, v2, device, method=fvd_method, only_final=True)
        out["fvd"] = {
            "value": float(res["value"][-1]),
            "reference": "gt" if track == "syn4d" else "source",
            "num_videos": v1.shape[0],
            "num_frames": int(T),
        }
    else:
        out["fvd"] = None
    return out
