"""The four task agents. Time and Location only *decide* values here; their action is applied inside the
stage owned by Style (global architectural language) and Context (background) respectively - except that
Location also owns the Context stage.  Actions are image operations, see ImageAgent._search.
"""
from .. import taxonomy as T
from .base import ImageAgent, State


def _prompt(st: State, only):
    return T.build_caption(__import__("archa.spec", fromlist=["x"]).spec_to_fields(st.spec), only=only)


class TimeAgent(ImageAgent):
    name = "time"
    fields = ["century_bin"]


class StyleAgent(ImageAgent):
    """Stage 1: sketch -> image. LoRA(style) at several strengths; pick the one the verifier likes."""
    name = "style"
    fields = ["style_category"]
    protect = []

    def act(self, st: State):
        g = self.ctx.generator
        prompt = _prompt(st, ["style", "century"])
        params = [{"lora": 1.0}, {"lora": 0.6}, {"lora": 1.4}][:self.budget]
        run = lambda p: g.generate(st.sketch, prompt, seed=st.seed, lora_scales={"style": p["lora"]})
        st.trace.append({"agent": self.name, "phase": "prompt", "prompt": prompt})
        return self._search(st, params, run, target=["style_category", "century_bin"])

    budget = 3


class MaterialAgent(ImageAgent):
    """Stage 2: re-generate the BUILDING region only (mask), low/medium noise, LoRA(material)."""
    name = "material"
    fields = ["materials", "color"]
    protect = ["style_category", "century_bin"]
    budget = 4

    def act(self, st: State):
        g = self.ctx.generator
        mask = self.ctx.masker(st.image)
        prompt = _prompt(st, ["materials", "color"])
        grid = [{"strength": s, "lora": l} for s in (0.55, 0.75) for l in (0.8, 1.2)][:self.budget]
        run = lambda p: g.edit(st.image, st.sketch, prompt, mask, strength=p["strength"], seed=st.seed,
                               lora_scales={"material": p["lora"]})
        st.trace.append({"agent": self.name, "phase": "prompt", "prompt": prompt})
        return self._search(st, grid, run, target=["materials", "color"])


class LocationAgent(ImageAgent):
    """Stage 3: re-generate the BACKGROUND only (inverse building mask), LoRA(context)."""
    name = "location"
    fields = ["continent"]
    protect = ["style_category", "century_bin", "materials", "color"]
    budget = 4

    def act(self, st: State):
        from PIL import Image, ImageFilter
        g = self.ctx.generator
        b = self.ctx.masker(st.image)
        grown = b.filter(ImageFilter.MaxFilter(9))                      # keep a margin around the building
        bg = Image.eval(grown, lambda v: 255 - v)
        prompt = _prompt(st, ["location", "environment"])
        grid = [{"strength": s, "lora": l} for s in (0.7, 0.9) for l in (0.8, 1.2)][:self.budget]
        run = lambda p: g.edit(st.image, st.sketch, prompt, bg, strength=p["strength"], seed=st.seed,
                               lora_scales={"context": p["lora"]})
        st.trace.append({"agent": self.name, "phase": "prompt", "prompt": prompt})
        return self._search(st, grid, run, target=["continent"])
