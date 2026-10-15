"""STEP 6 - building / background masks (used by the Material and Context LoRAs and agents).

Uses SegFormer fine-tuned on ADE20K. Mask value 255 = building (walls, windows, doors, towers ...),
0 = everything else (sky, trees, ground, people ...). Saved next to the cache: <cache>/masks/<id>.png

Run:  python scripts/06_make_masks.py --data_dir cache
Then LOOK at ~30 masks (cache/masks_preview/) before training anything on them.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from archa.data import read_jsonl  # noqa: E402

BUILDING_WORDS = ("building", "house", "skyscraper", "tower", "hovel", "wall", "windowpane",
                  "door", "column", "awning", "pier", "bridge", "stairway", "railing", "fence")


def building_ids(id2label):
    ids = []
    for i, name in id2label.items():
        if any(w == n.strip() or n.strip().startswith(w) for n in name.split(",") for w in BUILDING_WORDS):
            ids.append(int(i))
    return ids


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="cache")
    ap.add_argument("--model", default="nvidia/segformer-b2-finetuned-ade-512-512")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--preview", type=int, default=30)
    a = ap.parse_args()
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SegformerImageProcessor.from_pretrained(a.model)
    model = SegformerForSemanticSegmentation.from_pretrained(a.model).to(dev).eval()
    ids = building_ids(model.config.id2label)
    print("building classes:", [model.config.id2label[i] for i in ids])
    out_dir = os.path.join(a.data_dir, "masks")
    prev_dir = os.path.join(a.data_dir, "masks_preview")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)
    n_prev = 0
    for split in ("train", "val", "test"):
        rows = read_jsonl(os.path.join(a.data_dir, f"{split}.jsonl"))
        for i in tqdm(range(0, len(rows), a.batch), desc=split):
            chunk = rows[i:i + a.batch]
            imgs = [Image.open(r["image"]).convert("RGB") for r in chunk]
            x = proc(images=imgs, return_tensors="pt").to(dev)
            logits = model(**x).logits
            for r, im, lg in zip(chunk, imgs, logits):
                up = torch.nn.functional.interpolate(lg[None], size=im.size[::-1], mode="bilinear",
                                                     align_corners=False)[0]
                seg = up.argmax(0).cpu().numpy()
                m = np.isin(seg, ids).astype(np.uint8) * 255
                Image.fromarray(m).save(os.path.join(out_dir, r["id"] + ".png"))
                if n_prev < a.preview:
                    ov = np.array(im).copy()
                    ov[m == 0] = (ov[m == 0] * 0.3).astype(np.uint8)
                    Image.fromarray(ov).save(os.path.join(prev_dir, r["id"] + ".jpg"))
                    n_prev += 1
    print("masks ->", out_dir, "| previews ->", prev_dir)


if __name__ == "__main__":
    main()
