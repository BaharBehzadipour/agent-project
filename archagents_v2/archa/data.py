"""Torch datasets shared by ControlNet / ArchCLIP / LoRA training."""
import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import sketch as S
from . import taxonomy as T


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class ControlDataset(Dataset):
    """(photo, sketch, caption) triplets for ControlNet / LoRA training.

    train=True  : random crop, h-flip, sketch augmentation, attribute dropout
                  on the caption, occasional empty caption (for CFG).
    train=False : center crop, full caption.
    only_fields : restrict caption to these fields (per-task LoRAs).
    masks_dir   : optional dir with <id>.png building masks (255 = building);
                  returned as `loss_mask` for masked-loss LoRA training.
    """

    def __init__(self, jsonl, tokenizer, resolution=512, train=True, attr_drop=0.25,
                 empty_prob=0.1, only_fields=None, masks_dir=None, mask_region="building",
                 sketch_aug=True, require_known=False):
        rows = read_jsonl(jsonl)
        if require_known:
            rows = [r for r in rows if r["fields"]["known"]]
        self.rows, self.tok, self.res, self.train = rows, tokenizer, resolution, train
        self.attr_drop, self.empty_prob = attr_drop, empty_prob
        self.only, self.masks_dir, self.mask_region = only_fields, masks_dir, mask_region
        self.sketch_aug = sketch_aug and train

    def __len__(self):
        return len(self.rows)

    def _crop_params(self, w, h, rng):
        if self.train:
            return rng.randint(0, w - self.res), rng.randint(0, h - self.res)
        return (w - self.res) // 2, (h - self.res) // 2

    def __getitem__(self, i):
        rng = random.Random() if self.train else random.Random(i)
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        sk = Image.open(r["cond"]).convert("L")
        if sk.size != img.size:
            sk = sk.resize(img.size, Image.BOX)

        # short side -> resolution (sketch keeps its lines thanks to BOX averaging)
        w, h = img.size
        s = self.res / min(w, h)
        nw, nh = max(self.res, round(w * s)), max(self.res, round(h * s))
        img = img.resize((nw, nh), Image.BICUBIC)
        sk = sk.resize((nw, nh), Image.BOX)

        mask = None
        if self.masks_dir:
            mp = f"{self.masks_dir}/{r['id']}.png"
            m = Image.open(mp).convert("L").resize((nw, nh), Image.BILINEAR)
            mask = m

        x0, y0 = self._crop_params(nw, nh, rng)
        box = (x0, y0, x0 + self.res, y0 + self.res)
        img, sk = img.crop(box), sk.crop(box)
        if mask is not None:
            mask = mask.crop(box)
        if self.train and rng.random() < 0.5:
            img, sk = img.transpose(Image.FLIP_LEFT_RIGHT), sk.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.sketch_aug:
            if rng.random() < 0.3:
                sk = S.thicken(sk, 3)
            if rng.random() < 0.2:
                sk = S.random_erase_lines(sk, rng)

        px = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        cond = torch.from_numpy(np.array(sk.convert("RGB"))).permute(2, 0, 1).float() / 255.0

        cap = T.build_caption(
            r["fields"],
            drop_prob=self.attr_drop if self.train else 0.0,
            empty_prob=self.empty_prob if self.train else 0.0,
            rng=rng, only=self.only,
        )
        ids = self.tok(cap, max_length=self.tok.model_max_length, padding="max_length",
                       truncation=True, return_tensors="pt").input_ids[0]
        out = {"pixel_values": px, "conditioning_pixel_values": cond, "input_ids": ids,
               "caption": cap, "id": r["id"]}
        if mask is not None:
            m = torch.from_numpy(np.array(mask)).float() / 255.0
            out["loss_mask"] = m if self.mask_region == "building" else 1.0 - m
        return out


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in
           ("pixel_values", "conditioning_pixel_values", "input_ids")}
    if "loss_mask" in batch[0]:
        out["loss_mask"] = torch.stack([b["loss_mask"] for b in batch])
    out["caption"] = [b["caption"] for b in batch]
    out["id"] = [b["id"] for b in batch]
    return out
