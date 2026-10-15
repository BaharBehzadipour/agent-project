"""STEP 1 - clean the metadata, build captions, make grouped train/val split.

Inputs  : the two metadata files (train = 19,440 rows, test = 1,023 rows) and the
          folders that hold the images / sketches.
Outputs : <out_dir>/{train,val,test}.jsonl, taxonomy.json, stats.json, cooc.json

Run:
  python scripts/01_prepare_data.py \
      --train_meta  "/path/metadata (2).jsonl" --train_root /path/dataset/train \
      --test_meta   "/path/metadata (3).jsonl" --test_root  /path/dataset/test \
      --out_dir clean_data
Use --no_check_files to skip the file-existence check (e.g. on a laptop without images).
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from archa import taxonomy as T  # noqa: E402


def resolve(root, rel):
    """metadata says images/000029.jpg ; some layouts keep only the basename."""
    for cand in (os.path.join(root, rel), os.path.join(root, os.path.basename(os.path.dirname(rel)),
                                                       os.path.basename(rel))):
        if os.path.exists(cand):
            return cand
    return None


def load(meta_path, root, check_files):
    rows, missing = [], 0
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            cap = j.get("caption", {})
            img = os.path.join(root, j["file_name"])
            cond = os.path.join(root, j["conditioning_file_name"])
            if check_files:
                img, cond = resolve(root, j["file_name"]), resolve(root, j["conditioning_file_name"])
                if img is None or cond is None:
                    missing += 1
                    continue
            rec = T.clean_record(cap)
            title = T.clean_str(cap.get("title"))
            rows.append({
                "id": os.path.splitext(os.path.basename(j["file_name"]))[0],
                "image": img, "cond": cond,
                "title": title, "url": cap.get("url"),
                "fields": rec,
                "caption": T.build_caption(rec),
            })
    return rows, missing


def group_key(r, title_counts):
    t = r["title"]
    if t and title_counts[t.lower()] <= 20:     # generic titles ("Victorian House") are not groups
        return "t:" + t.lower()
    return "i:" + r["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_meta", required=True)
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--test_meta", required=True)
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--out_dir", default="clean_data")
    ap.add_argument("--val_size", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_check_files", action="store_true")
    ap.add_argument("--keep_placeholder_rows", action="store_true",
                    help="keep rows whose labels are template echoes (default: drop from train/val)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    train, m1 = load(a.train_meta, a.train_root, not a.no_check_files)
    test, m2 = load(a.test_meta, a.test_root, not a.no_check_files)
    print(f"loaded train={len(train)} (missing files {m1})  test={len(test)} (missing files {m2})")

    n_ph = sum(r["fields"]["placeholder_row"] for r in train)
    print(f"template-echo rows in train: {n_ph}")
    if not a.keep_placeholder_rows:
        train = [r for r in train if not r["fields"]["placeholder_row"]]
    for r in test:      # test is kept whole (comparable to the paper); flag only
        r["flag_placeholder"] = r["fields"]["placeholder_row"]

    # ---- grouped split (no building appears in both train and val) ----
    tc = collections.Counter((r["title"] or "").lower() for r in train if r["title"])
    groups = collections.defaultdict(list)
    for r in train:
        groups[group_key(r, tc)].append(r)
    keys = sorted(groups)
    random.Random(a.seed).shuffle(keys)
    val, tr = [], []
    for k in keys:
        (val if len(val) < a.val_size else tr).extend(groups[k])
    print(f"train={len(tr)} val={len(val)}")

    # ---- leakage report train vs test (same title) ----
    tr_titles = {(r["title"] or "").lower() for r in tr if r["title"]}
    leak = [r["id"] for r in test if r["title"] and r["title"].lower() in tr_titles
            and tc[r["title"].lower()] <= 20]
    print(f"test rows whose (non-generic) title also occurs in train: {len(leak)}")

    def dump(name, rows):
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("train", tr)
    dump("val", val)
    dump("test", test)

    # ---- taxonomy + stats + co-occurrence (used by the consistency agent) ----
    def field_counts(rows, key):
        c = collections.Counter()
        for r in rows:
            v = r["fields"][key]
            if isinstance(v, list):
                c.update(v)
            elif v is not None:
                c[v] += 1
        return dict(c.most_common())

    countries = field_counts(tr, "country")
    tax = {
        "style_category": T.STYLE_CATEGORIES,
        "century_bin": T.CENTURY_BINS,
        "materials": T.MATERIALS,
        "color": T.COLORS,
        "view": T.VIEWS,
        "continent": sorted(field_counts(tr, "continent")),
        "region": sorted(field_counts(tr, "region")),
        "country": [c for c, n in countries.items() if n >= 50],
    }
    json.dump(tax, open(os.path.join(a.out_dir, "taxonomy.json"), "w"), indent=1)

    stats = {k: field_counts(tr, k) for k in
             ["style_category", "century_bin", "continent", "color", "view"]}
    stats["materials"] = field_counts(tr, "materials")
    stats["n_known_true"] = sum(r["fields"]["known"] for r in tr)
    stats["counts"] = {"train": len(tr), "val": len(val), "test": len(test),
                       "placeholder_rows_dropped": n_ph, "title_overlap_test_train": len(leak)}
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=1)

    cooc = collections.defaultdict(collections.Counter)
    for r in tr:
        f = r["fields"]
        if f["style_category"] and f["century_bin"]:
            cooc["style_x_century"][f"{f['style_category']}|{f['century_bin']}"] += 1
        if f["style_category"] and f["continent"]:
            cooc["style_x_continent"][f"{f['style_category']}|{f['continent']}"] += 1
        if f["century_bin"] and f["continent"]:
            cooc["century_x_continent"][f"{f['century_bin']}|{f['continent']}"] += 1
        for m in f["materials"]:
            if f["style_category"]:
                cooc["style_x_material"][f"{f['style_category']}|{m}"] += 1
            if f["century_bin"]:
                cooc["century_x_material"][f"{f['century_bin']}|{m}"] += 1
            if f["continent"]:
                cooc["continent_x_material"][f"{f['continent']}|{m}"] += 1
    json.dump({k: dict(v) for k, v in cooc.items()},
              open(os.path.join(a.out_dir, "cooc.json"), "w"), indent=1)

    print("\nexample captions:")
    for r in random.Random(1).sample(tr, 5):
        print(" -", r["caption"])
    print(f"\nDone -> {a.out_dir}")


if __name__ == "__main__":
    main()
