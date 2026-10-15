"""ArchCLIP: CLIP (+LoRA) with one classification head per architectural attribute.

Two uses:
  * verifier / metric : photo-domain model scores a generated image per attribute
  * perception        : sketch-domain model predicts attribute distributions from the sketch
Both are the same class, trained twice (`--domain photo` / `--domain sketch`).
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as TV

from . import sketch as S

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

# single-label heads: name -> key in rec["fields"]
SINGLE = {"style_category": "style_category", "century_bin": "century_bin",
          "continent": "continent", "color": "color", "view": "view", "country": "country"}
MULTI = {"materials": "materials"}          # multi-label (BCE)


def label_maps(taxonomy: dict) -> dict:
    # a head with no classes (e.g. no country has >=50 samples in a tiny test set) gets one dummy class
    vals = lambda h: taxonomy[h] or ["__none__"]
    maps = {h: {v: i for i, v in enumerate(vals(h))} for h in SINGLE}
    maps.update({h: {v: i for i, v in enumerate(vals(h))} for h in MULTI})
    return maps


def encode_labels(fields: dict, maps: dict):
    """-> ({head: int or -1}, {head: float vector or None})"""
    single = {h: maps[h].get(fields.get(SINGLE[h]), -1) for h in SINGLE}
    multi = {}
    for h in MULTI:
        mats = fields.get(MULTI[h]) or []
        if mats:
            v = np.zeros(len(maps[h]), np.float32)
            for m in mats:
                if m in maps[h]:
                    v[maps[h][m]] = 1
            multi[h] = v
        else:
            multi[h] = None
    return single, multi


def make_transform(train: bool, domain: str, size=224):
    norm = TV.Compose([TV.ToTensor(), TV.Normalize(CLIP_MEAN, CLIP_STD)])
    if train:
        geo = TV.Compose([TV.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.33),
                                               interpolation=TV.InterpolationMode.BICUBIC),
                          TV.RandomHorizontalFlip()])
        if domain == "photo":
            geo = TV.Compose([geo, TV.ColorJitter(0.2, 0.2, 0.2, 0.02)])
    else:
        geo = TV.Compose([TV.Resize(size, interpolation=TV.InterpolationMode.BICUBIC),
                          TV.CenterCrop(size)])
    return TV.Compose([geo, norm])


class ArchCLIP(nn.Module):
    def __init__(self, backbone="openai/clip-vit-base-patch16", maps=None, lora_r=16):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPModel, CLIPTokenizer
        self.backbone_name = backbone
        self.tokenizer = CLIPTokenizer.from_pretrained(backbone)
        clip = CLIPModel.from_pretrained(backbone)
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                         target_modules=["q_proj", "v_proj"])
        self.clip = get_peft_model(clip, cfg)
        self.maps = maps
        self.size = clip.config.vision_config.image_size
        d = clip.config.projection_dim
        self.heads = nn.ModuleDict({h: nn.Linear(d, len(m)) for h, m in maps.items()})

    # --- features ---
    def image_feats(self, pixel_values):
        return self.clip.get_image_features(pixel_values=pixel_values)

    def text_feats(self, input_ids, attention_mask=None):
        return self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    def logits(self, feats):
        return {h: head(feats) for h, head in self.heads.items()}

    # --- io ---
    def save(self, path, extra=None):
        os.makedirs(path, exist_ok=True)
        self.clip.save_pretrained(path)
        torch.save(self.heads.state_dict(), os.path.join(path, "heads.pt"))
        json.dump({"backbone": self.backbone_name, "maps": self.maps, **(extra or {})},
                  open(os.path.join(path, "archclip.json"), "w"))

    @classmethod
    def load(cls, path, device="cpu"):
        from peft import PeftModel
        meta = json.load(open(os.path.join(path, "archclip.json")))
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        from transformers import CLIPModel, CLIPTokenizer
        m.backbone_name, m.maps = meta["backbone"], meta["maps"]
        m.tokenizer = CLIPTokenizer.from_pretrained(meta["backbone"])
        base = CLIPModel.from_pretrained(meta["backbone"])
        m.clip = PeftModel.from_pretrained(base, path)
        d = base.config.projection_dim
        m.heads = nn.ModuleDict({h: nn.Linear(d, len(mp)) for h, mp in m.maps.items()})
        m.heads.load_state_dict(torch.load(os.path.join(path, "heads.pt"), map_location="cpu"))
        m.size = base.config.vision_config.image_size
        m.domain = meta.get("domain", "photo")
        return m.to(device).eval()

    @torch.no_grad()
    def predict(self, images, domain=None, device=None):
        """list[PIL] -> {head: np.array [N, C]} (softmax / sigmoid probabilities)."""
        device = device or next(self.parameters()).device
        tf = make_transform(False, domain or getattr(self, "domain", "photo"), self.size)
        x = torch.stack([tf(im.convert("RGB")) for im in images]).to(device)
        lg = self.logits(self.image_feats(x))
        out = {}
        for h, v in lg.items():
            out[h] = (torch.sigmoid(v) if h in MULTI else torch.softmax(v, -1)).float().cpu().numpy()
        return out

    @torch.no_grad()
    def embed_images(self, images, domain=None, device=None):
        device = device or next(self.parameters()).device
        tf = make_transform(False, domain or getattr(self, "domain", "photo"), self.size)
        x = torch.stack([tf(im.convert("RGB")) for im in images]).to(device)
        return torch.nn.functional.normalize(self.image_feats(x), dim=-1)
