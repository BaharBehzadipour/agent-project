"""k-nearest-neighbour retrieval over the training sketches (sketch-domain ArchCLIP embeddings)."""
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .data import read_jsonl


@torch.no_grad()
def build_index(model, jsonl, out_npz, batch=64):
    rows = read_jsonl(jsonl)
    embs = []
    for i in tqdm(range(0, len(rows), batch), desc="index"):
        ims = [Image.open(r["cond"]).convert("RGB") for r in rows[i:i + batch]]
        embs.append(model.embed_images(ims, domain="sketch").cpu().numpy())
    np.savez(out_npz, emb=np.concatenate(embs), ids=np.array([r["id"] for r in rows]))


class Retriever:
    def __init__(self, npz_path, jsonl):
        z = np.load(npz_path)
        self.emb = z["emb"].astype(np.float32)
        rows = {r["id"]: r for r in read_jsonl(jsonl)}
        self.rows = [rows[i] for i in z["ids"]]

    def query(self, e, k=5, exclude_id=None):
        sims = self.emb @ e.astype(np.float32)
        out = []
        for i in np.argsort(-sims):
            if exclude_id and self.rows[i]["id"] == exclude_id:
                continue
            out.append((self.rows[i], float(sims[i])))
            if len(out) == k:
                break
        return out

    @staticmethod
    def summarise(neigh):
        keys = ("style_category", "century_bin", "continent", "materials", "color")
        return [{k: r["fields"].get(k) for k in keys} for r, _ in neigh]
