"""STEP 9 - run the agent pipeline on ONE sketch.

  python scripts/09_run_agents.py --sketch my_sketch.png --out_dir results/demo
  python scripts/09_run_agents.py --sketch my_sketch.png --lock style_category=Gothic continent=Europe \
         materials=stone,glass --out_dir results/gothic --no_llm

Outputs in out_dir: sketch.png, stage_style.png, stage_material.png, stage_location.png, final.png,
spec_and_trace.json (every agent decision + every candidate + gate result).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image  # noqa: E402

from archa import archclip as A  # noqa: E402
from archa.pipeline import ArchPipeline, add_common_args, build_context  # noqa: E402
from archa.retrieval import Retriever  # noqa: E402

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--sketch", required=True, help="black lines on white background")
ap.add_argument("--lock", nargs="*", default=[], help="field=value (materials=a,b)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_dir", default="results/demo")
ap.add_argument("--stages", nargs="+", default=["style", "material", "location"])
a = ap.parse_args()

locked = {}
for kv in a.lock:
    k, v = kv.split("=", 1)
    locked[k] = v.split(",") if k == "materials" else v

ctx, dev = build_context(a)
sk_model = A.ArchCLIP.load(a.archclip_sketch, dev)
ret = Retriever(a.index, os.path.join(a.data_dir, "train.jsonl")) if os.path.exists(a.index) else None
pipe = ArchPipeline(ctx, sk_model, ret)
st = pipe.run(Image.open(a.sketch).convert("L"), locked, a.seed, a.out_dir, tuple(a.stages))
print("spec:", st.spec)
print("saved ->", a.out_dir)
