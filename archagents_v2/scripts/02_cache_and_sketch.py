"""STEP 2 - make a fast, uniform cache of images + sketches.

* resizes every image so its short side = --size (default 640)
* sketch: --mode provided      -> keep the paper's sketches (resized without losing lines)
          --mode adaptive_canny -> regenerate stronger sketches from the photo (recommended)
* writes new jsonl files (same fields, `image` / `cond` stored RELATIVE to the cache folder,
  so you can copy the folder to Drive / Colab and keep using it)

Run (after step 1):
  python scripts/02_cache_and_sketch.py --in_dir clean_data --out_dir cache --mode adaptive_canny
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image  # noqa: E402

from archa import sketch as S  # noqa: E402


def work(args):
    r, out_dir, size, mode = args
    img = Image.open(r["image"]).convert("RGB")
    img = S.resize_short_side(img, size, Image.LANCZOS) if min(img.size) > size else img
    if mode == "provided":
        sk = S.resize_sketch(Image.open(r["cond"]), min(img.size))
        if sk.size != img.size:
            sk = sk.resize(img.size, Image.BOX)
    else:
        sk = S.make_sketch(img, mode)
    ip = os.path.join(out_dir, "images", r["id"] + ".jpg")
    cp = os.path.join(out_dir, "sketches", r["id"] + ".png")
    img.save(ip, quality=95)
    sk.save(cp)
    r = dict(r)
    r["image"] = os.path.join("images", r["id"] + ".jpg")      # relative -> the cache folder is portable
    r["cond"] = os.path.join("sketches", r["id"] + ".png")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="clean_data")
    ap.add_argument("--out_dir", default="cache")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--mode", default="adaptive_canny", choices=["provided", "adaptive_canny"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "sketches"), exist_ok=True)
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(os.path.join(a.in_dir, f"{split}.jsonl"), encoding="utf-8")]
        jobs = [(r, a.out_dir, a.size, a.mode) for r in rows]
        with ProcessPoolExecutor(a.workers) as ex:
            out = []
            for i, r in enumerate(ex.map(work, jobs, chunksize=16)):
                out.append(r)
                if i % 1000 == 0:
                    print(f"{split}: {i}/{len(rows)}", flush=True)
        with open(os.path.join(a.out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: wrote {len(out)}")
    json.dump({"mode": a.mode, "size": a.size}, open(os.path.join(a.out_dir, "cache_info.json"), "w"))
    for f in ("taxonomy.json", "stats.json", "cooc.json"):
        src = os.path.join(a.in_dir, f)
        if os.path.exists(src):
            open(os.path.join(a.out_dir, f), "w").write(open(src).read())


if __name__ == "__main__":
    main()
