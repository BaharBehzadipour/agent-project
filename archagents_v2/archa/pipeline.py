"""ArchPipeline: perceive the sketch, let the agents decide in dependency order, then act in image stages.

 decide order : Time -> Location -> Style -> Material   (later ones are conditioned on earlier ones)
 act order    : Style(+Time) generate -> Material edit (building) -> Location edit (background)
"""
import json
import os

from PIL import Image

from . import archclip as A
from .agents.agents import LocationAgent, MaterialAgent, StyleAgent, TimeAgent
from .agents.base import Context, State
from .generator import BuildingMasker, Generator
from .llm import GeminiVLM
from .retrieval import Retriever
from .spec import Consistency
from .verifier import Verifier


def build_context(a, need_generator=True):
    """a: argparse namespace with the paths below."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tax = json.load(open(os.path.join(a.data_dir, "taxonomy.json")))
    ctx = Context(taxonomy=tax,
                  consistency=Consistency(os.path.join(a.data_dir, "cooc.json")),
                  use_gate=not a.no_gate, use_consistency=not a.no_consistency)
    ctx.llm = None if a.no_llm else GeminiVLM()
    if need_generator:
        loras = {n: p for n, p in (("style", a.lora_style), ("material", a.lora_material),
                                   ("context", a.lora_context)) if p}
        ctx.generator = Generator(a.controlnet, a.base_model, loras, dev, steps=a.steps, short=a.gen_short)
        ctx.masker = BuildingMasker(device=dev)
        ctx.verifier = Verifier(a.archclip_photo, dev)
    return ctx, dev


class ArchPipeline:
    def __init__(self, ctx, sketch_model, retriever=None):
        self.ctx, self.sk, self.ret = ctx, sketch_model, retriever
        self.time, self.loc = TimeAgent(ctx), LocationAgent(ctx)
        self.style, self.mat = StyleAgent(ctx), MaterialAgent(ctx)

    def perceive(self, sketch, locked, seed):
        st = State(sketch=sketch, seed=seed, locked=dict(locked))
        st.sketch_probs = {h: v[0] for h, v in self.sk.predict([sketch], domain="sketch").items()}
        if self.ret is not None:
            e = self.sk.embed_images([sketch], domain="sketch")[0].cpu().numpy()
            st.neighbours = Retriever.summarise(self.ret.query(e, 5))
        return st

    def decide(self, st):
        for ag in (self.time, self.loc, self.style, self.mat):
            ag.decide(st)
        st.trace.append({"phase": "consistency", "conflicts": self.ctx.consistency.conflicts(st.spec)})
        return st

    def run(self, sketch: Image.Image, locked=None, seed=0, out_dir=None, stages=("style", "material", "location")):
        st = self.decide(self.perceive(sketch, locked or {}, seed))
        stage_agents = {"style": self.style, "material": self.mat, "location": self.loc}
        for name in stages:
            st.image = stage_agents[name].act(st)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                st.image.save(os.path.join(out_dir, f"stage_{name}.png"))
        if out_dir:
            sketch.save(os.path.join(out_dir, "sketch.png"))
            st.image.save(os.path.join(out_dir, "final.png"))
            json.dump({"spec": st.spec, "locked": st.locked, "seed": seed, "trace": st.trace},
                      open(os.path.join(out_dir, "spec_and_trace.json"), "w"), indent=1, default=str)
        return st


def add_common_args(ap):
    ap.add_argument("--data_dir", default="cache")
    ap.add_argument("--controlnet", default="runs/cn_v2/last")
    ap.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--lora_style", default="runs/lora_style")
    ap.add_argument("--lora_material", default="runs/lora_material")
    ap.add_argument("--lora_context", default="runs/lora_context")
    ap.add_argument("--archclip_photo", default="runs/archclip_photo/best")
    ap.add_argument("--archclip_sketch", default="runs/archclip_sketch/best")
    ap.add_argument("--index", default="runs/retrieval_index.npz")
    ap.add_argument("--steps", type=int, default=28, help="diffusion steps per generation")
    ap.add_argument("--gen_short", type=int, default=512, help="short side of generated images")
    ap.add_argument("--no_llm", action="store_true", help="no VLM (ArchCLIP + consistency prior only)")
    ap.add_argument("--no_gate", action="store_true", help="ablation: disable non-regression gate")
    ap.add_argument("--no_consistency", action="store_true", help="ablation: disable consistency prior")
    return ap
