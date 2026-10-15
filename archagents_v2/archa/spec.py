"""Design spec (the blackboard the agents share) + dataset-derived consistency prior."""
import json
import math

import numpy as np

from . import archclip as A
from . import taxonomy as T

# representative century for each bin, used when an agent has to write it in a prompt
BIN_TO_CENTURY = {"ancient": 0, "medieval": 11, "c15-17": 16, "c18": 18, "c19": 19, "c20": 20, "c21": 21}


def spec_to_fields(spec: dict) -> dict:
    """spec = {head: value | [materials]}  ->  rec-like dict usable by taxonomy.build_caption."""
    return {
        "style_category": spec.get("style_category"),
        "style": (spec.get("style_category") or "").lower() if spec.get("style_category") else None,
        "century": BIN_TO_CENTURY.get(spec.get("century_bin")),
        "continent": spec.get("continent"),
        "country": spec.get("country"),
        "region": None,
        "materials": spec.get("materials") or [],
        "color": spec.get("color"),
        "view": spec.get("view"),
        "environment": spec.get("environment"),
    }


# pairs available in cooc.json:  key -> (field_a, field_b)  stored as "a|b"
_PAIRS = {
    "style_x_century": ("style_category", "century_bin"),
    "style_x_continent": ("style_category", "continent"),
    "century_x_continent": ("century_bin", "continent"),
    "style_x_material": ("style_category", "materials"),
    "century_x_material": ("century_bin", "materials"),
    "continent_x_material": ("continent", "materials"),
}


class Consistency:
    """P(value | already-decided fields) from dataset co-occurrence counts (add-one smoothing)."""

    def __init__(self, cooc_path):
        self.cooc = json.load(open(cooc_path))
        self.index = {}   # (field, other_field) -> {other_value: {value: count}}
        for key, (fa, fb) in _PAIRS.items():
            tab = self.cooc.get(key, {})
            for k, n in tab.items():
                a, b = k.split("|")
                self.index.setdefault((fa, fb), {}).setdefault(b, {})[a] = n
                self.index.setdefault((fb, fa), {}).setdefault(a, {})[b] = n

    def prior(self, field, candidates, spec, alpha=1.0):
        """-> np.array over `candidates` (normalised); uniform when nothing to condition on."""
        logp = np.zeros(len(candidates))
        for other, val in spec.items():
            if other == field or not val:
                continue
            tab = self.index.get((field, other))
            if not tab:
                continue
            others = val if isinstance(val, list) else [val]
            for o in others:
                row = tab.get(o)
                if not row:
                    continue
                tot = sum(row.get(c, 0) + 1 for c in candidates)
                logp += alpha * np.array([math.log((row.get(c, 0) + 1) / tot) for c in candidates])
        p = np.exp(logp - logp.max())
        return p / p.sum()

    def conflicts(self, spec, min_cond_prob=0.01):
        """Pairs whose observed co-occurrence is extremely rare -> [(fieldA, a, fieldB, b, p)]."""
        out = []
        for key, (fa, fb) in _PAIRS.items():
            tab = self.cooc.get(key, {})
            va, vb = spec.get(fa), spec.get(fb)
            for a in ([va] if isinstance(va, str) else (va or [])):
                for b in ([vb] if isinstance(vb, str) else (vb or [])):
                    tot_b = sum(n for k, n in tab.items() if k.split("|")[1] == b) + 1
                    p = (tab.get(f"{a}|{b}", 0) + 1) / tot_b
                    if p < min_cond_prob and tot_b > 30:
                        out.append((fa, a, fb, b, p))
        return out
