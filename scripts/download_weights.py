#!/usr/bin/env python3
"""Download the model weights the bench needs (VGGT-Omega checkpoint + I3D).

CLIP (ViT-H-14) and VBench models auto-download on first eval — nothing to fetch here.

    python scripts/download_weights.py
"""
from vdm_nvs_bench.weights import download_all

if __name__ == "__main__":
    download_all()
