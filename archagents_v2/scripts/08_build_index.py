"""STEP 8 - build the retrieval index (sketch embeddings of the training set) used by the agents.

  python scripts/08_build_index.py --data_dir cache --archclip_sketch runs/archclip_sketch/best
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402

from archa import archclip as A  # noqa: E402
from archa.retrieval import build_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="cache")
ap.add_argument("--archclip_sketch", default="runs/archclip_sketch/best")
ap.add_argument("--out", default="runs/retrieval_index.npz")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
build_index(A.ArchCLIP.load(a.archclip_sketch, dev), os.path.join(a.data_dir, "train.jsonl"), a.out)
print("index ->", a.out)
