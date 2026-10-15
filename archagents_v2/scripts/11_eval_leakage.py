"""STEP 11 - cross-attribute leakage: does changing ONE attribute disturb the others?

For N test sketches: fix sketch + seed, generate with the true spec (base), then change ONE field
(style / century / continent / materials) and regenerate. With the photo ArchCLIP heads we measure
  diagonal   : gain of the NEW value of the changed field (should be high)
  off-diagonal: mean |change| of the probability of the OTHER fields' base values (should be low)
for two systems: `monolithic` (one prompt, ControlNet only) and `agents` (staged, masked, gated).

  python scripts/11_eval_leakage.py --n 100 --no_llm
Outputs results/leakage.json and results/leakage.png
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from archa import archclip as A  # noqa: E402
from archa import taxonomy as T  # noqa: E402
from archa.data import read_jsonl  # noqa: E402
from archa.pipeline import ArchPipeline, add_common_args, build_context  # noqa: E402
from archa.spec import spec_to_fields  # noqa: E402

ap = add_common_args(argparse.ArgumentParser())
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_dir", default="results")
a = ap.parse_args()
FIELDS = ["style_category", "century_bin", "continent", "materials"]
rng = random.Random(a.seed)

ctx, dev = build_context(a)
sk_model = A.ArchCLIP.load(a.archclip_sketch, dev)
pipe = ArchPipeline(ctx, sk_model, None)

rows = [r for r in read_jsonl(os.path.join(a.data_dir, f"{a.split}.jsonl"))
        if not r.get("flag_placeholder") and all(r["fields"].get(f) for f in FIELDS)]
rng.shuffle(rows)
rows = rows[:a.n]
tax = ctx.taxonomy


def true_spec(r):
    f = r["fields"]
    return {"style_category": f["style_category"], "century_bin": f["century_bin"],
            "continent": f["continent"], "materials": f["materials"][:2], "color": f["color"] or "gray"}


def alt(field, spec):
    cur = spec[field]
    if field == "materials":
        pool = [m for m in tax["materials"] if m not in cur]
        return [rng.choice(pool)]
    pool = [v for v in tax[field] if v != cur]
    return rng.choice(pool)


def monolithic(sketch, spec, seed):
    prompt = T.build_caption(spec_to_fields(spec))
    return ctx.generator.generate(sketch, prompt, seed=seed, lora_scales={})


def agents(sketch, spec, seed):
    return pipe.run(sketch, dict(spec), seed).image


def prob_of(img, field, val):
    return ctx.verifier.score(img, {field: val}, [field]).get(field, 0.0)


systems = {"monolithic": monolithic, "agents": agents}
acc = {s: {f: {h: [] for h in FIELDS} for f in FIELDS} for s in systems}
for r in tqdm(rows):
    sketch = Image.open(r["cond"]).convert("L")
    base_spec = true_spec(r)
    for sname, fn in systems.items():
        base = fn(sketch, base_spec, a.seed)
        for f in FIELDS:
            new_spec = dict(base_spec)
            new_spec[f] = alt(f, base_spec)
            var = fn(sketch, new_spec, a.seed)
            for h in FIELDS:
                if h == f:    # gain of the new value
                    acc[sname][f][h].append(prob_of(var, h, new_spec[h]) - prob_of(base, h, new_spec[h]))
                else:         # disturbance of an untouched attribute
                    acc[sname][f][h].append(abs(prob_of(var, h, base_spec[h]) - prob_of(base, h, base_spec[h])))
res = {s: {f: {h: float(np.mean(v)) for h, v in d.items()} for f, d in m.items()} for s, m in acc.items()}
os.makedirs(a.out_dir, exist_ok=True)
json.dump(res, open(os.path.join(a.out_dir, "leakage.json"), "w"), indent=1)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, (s, m) in enumerate(res.items()):
        M = np.array([[m[f][h] for h in FIELDS] for f in FIELDS])
        im = ax[k].imshow(M, cmap="viridis")
        ax[k].set_xticks(range(len(FIELDS)), FIELDS, rotation=30)
        ax[k].set_yticks(range(len(FIELDS)), ["change " + f for f in FIELDS])
        ax[k].set_title(s)
        for i in range(len(FIELDS)):
            for j in range(len(FIELDS)):
                ax[k].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.suptitle("diagonal = gain of changed attribute (high is good); off-diagonal = leakage (low is good)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "leakage.png"), dpi=150)
except Exception as e:
    print("plot skipped:", e)
for s, m in res.items():
    diag = np.mean([m[f][f] for f in FIELDS])
    off = np.mean([m[f][h] for f in FIELDS for h in FIELDS if h != f])
    print(f"{s}: mean gain (diag) {diag:.3f} | mean leakage (off-diag) {off:.3f}")
