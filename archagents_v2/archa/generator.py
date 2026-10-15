"""Image actuators: SD-1.5 + the new ControlNet + switchable task LoRAs.

`generate`  : sketch -> image (stage 1)
`edit`      : masked, strength-controlled re-generation of part of an existing image (stages 2, 3)
The geometry is ALWAYS pinned by the sketch through the (frozen) ControlNet.
"""
import numpy as np
import torch
from PIL import Image, ImageFilter


def fit_size(w, h, short=512, max_long=768):
    s = short / min(w, h)
    nw, nh = w * s, h * s
    if max(nw, nh) > max_long:
        k = max_long / max(nw, nh)
        nw, nh = nw * k, nh * k
    return int(round(nw / 8) * 8), int(round(nh / 8) * 8)


class Generator:
    def __init__(self, controlnet, base_model="stable-diffusion-v1-5/stable-diffusion-v1-5", loras=None,
                 device=None, steps=28, guidance=7.0, short=512):
        from diffusers import (ControlNetModel, StableDiffusionControlNetInpaintPipeline,
                               StableDiffusionControlNetPipeline, UniPCMultistepScheduler)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if self.device == "cuda" else torch.float32
        cn = ControlNetModel.from_pretrained(controlnet, torch_dtype=dt)
        self.t2i = StableDiffusionControlNetPipeline.from_pretrained(
            base_model, controlnet=cn, torch_dtype=dt, safety_checker=None).to(self.device)
        self.t2i.scheduler = UniPCMultistepScheduler.from_config(self.t2i.scheduler.config)
        # inpaint pipeline SHARES unet/vae/controlnet with t2i, so LoRAs loaded below apply to both
        self.inp = StableDiffusionControlNetInpaintPipeline(**self.t2i.components)
        self.inp.scheduler = self.t2i.scheduler
        for p in (self.t2i, self.inp):
            p.set_progress_bar_config(disable=True)
        self.adapters = []
        for name, path in (loras or {}).items():
            self.t2i.load_lora_weights(path, adapter_name=name)
            self.adapters.append(name)
        self.steps, self.guidance, self.short = steps, guidance, short

    def _set_adapters(self, scales):
        scales = scales or {}
        if not self.adapters:
            return
        names = list(self.adapters)
        self.t2i.set_adapters(names, adapter_weights=[float(scales.get(n, 0.0)) for n in names])

    def _gen(self, seed):
        return torch.Generator(device=self.device).manual_seed(int(seed))

    @torch.no_grad()
    def generate(self, sketch, prompt, seed=0, lora_scales=None, negative="lowres, blurry, text, watermark",
                 control_scale=1.0):
        size = fit_size(*sketch.size, short=self.short, max_long=int(self.short * 1.5))
        sk = sketch.convert("RGB").resize(size, Image.BOX)
        self._set_adapters(lora_scales)
        return self.t2i(prompt, negative_prompt=negative, image=sk, width=size[0], height=size[1],
                        num_inference_steps=self.steps, guidance_scale=self.guidance,
                        controlnet_conditioning_scale=control_scale, generator=self._gen(seed)).images[0]

    @torch.no_grad()
    def edit(self, image, sketch, prompt, mask, strength=0.7, seed=0, lora_scales=None,
             negative="lowres, blurry, text, watermark", control_scale=1.0, feather=4):
        """Re-generate only where mask==255, starting from `image` at the given noise strength."""
        size = image.size
        sk = sketch.convert("RGB").resize(size, Image.BOX)
        m = mask.convert("L").resize(size, Image.BILINEAR)
        if feather:
            m = m.filter(ImageFilter.GaussianBlur(feather))
        self._set_adapters(lora_scales)
        out = self.inp(prompt, negative_prompt=negative, image=image.convert("RGB"), mask_image=m,
                       control_image=sk, width=size[0], height=size[1], strength=float(strength),
                       num_inference_steps=self.steps, guidance_scale=self.guidance,
                       controlnet_conditioning_scale=control_scale, generator=self._gen(seed)).images[0]
        # keep the untouched area pixel-identical
        keep = np.array(m).astype(np.float32)[..., None] / 255.0
        res = np.array(out).astype(np.float32) * keep + np.array(image.convert("RGB")).astype(np.float32) * (1 - keep)
        return Image.fromarray(res.round().astype(np.uint8))


class BuildingMasker:
    """SegFormer-ADE20K building mask for an arbitrary image (same classes as step 6)."""

    def __init__(self, model="nvidia/segformer-b2-finetuned-ade-512-512", device=None):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = SegformerImageProcessor.from_pretrained(model)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model).to(self.device).eval()
        import importlib.util, os, sys
        spec = importlib.util.spec_from_file_location(
            "mk", os.path.join(os.path.dirname(__file__), "..", "scripts", "06_make_masks.py"))
        mk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mk)
        self.ids = mk.building_ids(self.model.config.id2label)

    @torch.no_grad()
    def __call__(self, image):
        x = self.proc(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        lg = self.model(**x).logits
        up = torch.nn.functional.interpolate(lg, size=image.size[::-1], mode="bilinear", align_corners=False)[0]
        seg = up.argmax(0).cpu().numpy()
        return Image.fromarray((np.isin(seg, self.ids).astype(np.uint8)) * 255)
