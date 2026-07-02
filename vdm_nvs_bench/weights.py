"""Weight resolution / download for the self-contained bench.

- VGGT-Omega checkpoint (~4.6 GB): HF `facebook/VGGT-Omega` -> weights/.
- I3D torchscript (~49 MB): the styleganv FVD weight, fetched to the vendored
  common_metrics/fvd/styleganv/ (also auto-wget'd by the FVD code as a fallback).
- CLIP (ViT-H-14) and VBench models auto-download on first use — nothing to do here.

Override the VGGT-Omega checkpoint with env VGGT_OMEGA_CKPT or --checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "weights"
VGGT_HF_REPO = "facebook/VGGT-Omega"
VGGT_CKPT_NAME = "vggt_omega_1b_512.pt"
I3D_PATH = REPO_ROOT / "vdm_nvs_bench" / "video" / "common_metrics" / "fvd" / "styleganv" / "i3d_torchscript.pt"
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"


def resolve_vggt_checkpoint(download: bool = True) -> Path:
    """Return a local VGGT-Omega checkpoint path, downloading from HF if needed."""
    env = os.environ.get("VGGT_OMEGA_CKPT")
    if env and Path(env).is_file():
        return Path(env)
    local = WEIGHTS_DIR / VGGT_CKPT_NAME
    if local.is_file():
        return local
    if not download:
        raise FileNotFoundError(
            f"VGGT-Omega checkpoint not found ({local}). Run scripts/download_weights.py "
            f"or set VGGT_OMEGA_CKPT."
        )
    from huggingface_hub import hf_hub_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[weights] downloading {VGGT_HF_REPO}/{VGGT_CKPT_NAME} -> {WEIGHTS_DIR}")
    path = hf_hub_download(repo_id=VGGT_HF_REPO, filename=VGGT_CKPT_NAME, local_dir=str(WEIGHTS_DIR))
    return Path(path)


def ensure_i3d() -> Path:
    """Ensure the styleganv I3D weight is present (download if missing)."""
    if I3D_PATH.is_file() and I3D_PATH.stat().st_size > 1_000_000:
        return I3D_PATH
    import urllib.request

    I3D_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[weights] downloading I3D -> {I3D_PATH}")
    urllib.request.urlretrieve(I3D_URL, I3D_PATH)
    return I3D_PATH


def download_all() -> None:
    ensure_i3d()
    resolve_vggt_checkpoint(download=True)
    print("[weights] done. CLIP + VBench models auto-download on first eval.")


if __name__ == "__main__":
    download_all()
