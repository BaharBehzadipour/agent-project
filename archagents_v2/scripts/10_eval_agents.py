"""STEP 10 - evaluate the agents' DECISIONS (no image generation, runs in minutes).

For each val/test sketch the agents predict style / century / continent / materials / color from the
SKETCH ONLY; we compare with the metadata. Variants: ArchCLIP argmax | + consistency prior | + VLM.

  python scripts/10_eval_agents.py --split test --no_llm
  GEMINI_API_KEY=... python scripts/10_eval_agents.py --split test --limit 200
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from archa import archclip as A  # noqa: E402
from archa.data import read_jsonl  # noqa: E402
from archa.llm import GeminiVLM  # noqa: E402
from archa.pipeline import ArchPipeline, add_common_args, build_context  # noqa: E402
from archa.retrieval import Retriever  # noqa: E402

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--split", default="test", choices=["val", "test"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only_known", action="store_true")
ap.add_argument("--out", default="results/agent_decisions.json")
a = ap.parse_args()

ctx, dev = build_context(a, need_generator=False)
rows = read_jsonl(os.path.join(a.data_dir, f"{a.split}.jsonl"))
rows = [r for r in rows if not r.get("flag_placeholder") and (r["fields"]["known"] or not a.only_known)]
if a.limit:
    rows = rows[:a.limit]
sk = A.ArchCLIP.load(a.archclip_sketch, dev)
ret = Retriever(a.index, os.path.join(a.data_dir, "train.jsonl")) if os.path.exists(a.index) else None
pipe = ArchPipeline(ctx, sk, ret)
HEADS = ["style_category", "century_bin", "continent", "materials", "color"]

variants = {"archclip": dict(use_consistency=False, llm=None),
            "archclip+prior": dict(use_consistency=True, llm=None)}
if not a.no_llm:
    variants["agents(+vlm)"] = dict(use_consistency=True, llm=GeminiVLM())

def f1_set(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred or not gold:
        return None
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))

res = {}
for name, cfg in variants.items():
    ctx.use_consistency, ctx.llm = cfg["use_consistency"], cfg["llm"]
    hits = {h: [] for h in HEADS}
    for r in tqdm(rows, desc=name):
        st = pipe.decide(pipe.perceive(Image.open(r["cond"]).convert("L"), {}, 0))
        for h in HEADS:
            gold = r["fields"].get(h)
            if gold in (None, [], ""):
                continue
            hits[h].append(f1_set(st.spec[h], gold) if h == "materials" else float(st.spec[h] == gold))
    res[name] = {h: float(np.mean([x for x in v if x is not None])) for h, v in hits.items() if v}
    res[name]["n"] = len(rows)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print("\n| variant | " + " | ".join(HEADS) + " |\n|---|" + "---|" * len(HEADS))
for n, r in res.items():
    print(f"| {n} | " + " | ".join(f"{r.get(h, float('nan')):.3f}" for h in HEADS) + " |")
