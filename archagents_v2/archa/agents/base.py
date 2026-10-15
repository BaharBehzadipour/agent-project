"""Agent base class: perceive -> decide -> act (image op) -> verify, with a non-regression gate.

An agent OWNS some spec fields (`fields`). It never edits another agent's fields. Its action is an
image operation (generate / masked edit) controlled by numeric parameters, and every candidate result is
checked by the photo ArchCLIP: the target fields must not get worse and the protected (earlier) fields
must not drop by more than `eps`.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image

from .. import archclip as A
from ..spec import spec_to_fields


@dataclass
class State:
    sketch: Image.Image
    seed: int = 0
    locked: dict = field(default_factory=dict)      # user-given fields (never changed)
    spec: dict = field(default_factory=dict)        # the shared blackboard
    image: Optional[Image.Image] = None             # current image
    sketch_probs: dict = field(default_factory=dict)
    neighbours: list = field(default_factory=list)
    trace: list = field(default_factory=list)


@dataclass
class Context:
    taxonomy: dict
    consistency: Any
    verifier: Any = None
    generator: Any = None
    masker: Any = None
    llm: Any = None
    use_gate: bool = True
    use_consistency: bool = True
    eps: float = 0.08
    topk: int = 4


class ImageAgent:
    name = "agent"
    fields: list = []          # spec heads this agent decides
    protect: list = []         # fields (from earlier stages) the action must not damage

    def __init__(self, ctx: Context):
        self.ctx = ctx

    # ---------------- decide ----------------
    def decide(self, st: State):
        for head in self.fields:
            if head in st.locked:
                st.spec[head] = st.locked[head]
                st.trace.append({"agent": self.name, "phase": "decide", "field": head,
                                 "value": st.locked[head], "source": "locked"})
                continue
            self._decide_head(st, head)

    def _decide_head(self, st, head):
        names = self.ctx.taxonomy[head]
        p = np.asarray(st.sketch_probs[head], dtype=np.float64)
        multi = head in A.MULTI
        prior = (self.ctx.consistency.prior(head, names, st.spec, alpha=0.7)
                 if self.ctx.use_consistency else np.ones(len(names)) / len(names))
        score = p * (prior * len(names)) ** (0.7 if not multi else 0.5)
        order = np.argsort(-score)
        cands = [(names[i], float(p[i])) for i in order[:self.ctx.topk]]
        source = "archclip+prior" if self.ctx.use_consistency else "archclip"
        llm_val = None
        if self.ctx.llm is not None and getattr(self.ctx.llm, "available", False):
            try:
                llm_val = self.ctx.llm.choose(head, cands, st.sketch, st.spec, st.neighbours, multi=multi)
            except Exception as e:                     # network / quota -> fall back silently
                st.trace.append({"agent": self.name, "phase": "decide", "field": head,
                                 "llm_error": str(e)[:200]})
        if multi:
            if llm_val:
                val = llm_val[:3]
                source = "vlm"
            else:
                top = score[order[0]]
                val = [names[i] for i in order[:3] if score[i] >= 0.5 * top and p[i] > 0.25] or [names[order[0]]]
        else:
            val = llm_val if llm_val else names[order[0]]
            source = "vlm" if llm_val else source
        st.spec[head] = val
        st.trace.append({"agent": self.name, "phase": "decide", "field": head, "value": val,
                         "candidates": cands, "source": source})

    # ---------------- act ----------------
    def act(self, st: State):
        return st.image

    # -------- shared helper: try candidate parameters, keep the best that passes the gate --------
    def _search(self, st: State, params: list, run, target: list):
        v = self.ctx.verifier
        prot = [f for f in self.protect if f in st.spec]
        base_prot = v.score(st.image, st.spec, prot) if (st.image is not None and prot) else {}
        base_tgt = v.score(st.image, st.spec, target) if st.image is not None else {}
        base_t = float(np.mean(list(base_tgt.values()))) if base_tgt else 0.0
        best, rows = None, []
        for prm in params:
            img = run(prm)
            tg = v.score(img, st.spec, target)
            pr = v.score(img, st.spec, prot) if prot else {}
            t = float(np.mean(list(tg.values()))) if tg else 0.0
            drop = max([base_prot.get(k, 0) - pr.get(k, 0) for k in base_prot] or [0.0])
            ok = (drop <= self.ctx.eps) if self.ctx.use_gate else True
            better = (t >= base_t - 1e-6) if st.image is not None else True
            rows.append({"params": prm, "target": tg, "target_mean": t, "protected_drop": drop,
                         "gate_ok": bool(ok and better)})
            key = (ok and better, t - 0.5 * max(drop, 0.0))
            if best is None or key > best[0]:
                best = (key, img, prm)
        accepted = best[0][0]
        st.trace.append({"agent": self.name, "phase": "act", "base_target": base_t, "candidates": rows,
                         "accepted": bool(accepted), "chosen": best[2] if accepted else None})
        if accepted:
            return best[1]
        return st.image if st.image is not None else best[1]   # stage 1 always needs an image

    def fields_dict(self, st):
        return spec_to_fields(st.spec)
