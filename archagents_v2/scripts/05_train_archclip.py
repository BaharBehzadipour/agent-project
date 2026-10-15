"""STEP 5 - train ArchCLIP (attribute heads + contrastive caption loss).

Run it TWICE:
  python scripts/05_train_archclip.py --data_dir cache --domain photo  --out_dir runs/archclip_photo
  python scripts/05_train_archclip.py --data_dir cache --domain sketch --out_dir runs/archclip_sketch

photo  model : verifier / evaluation metric on generated images
sketch model : perception (predicts attributes from the user's sketch) and retrieval
Training uses ONLY train.jsonl; model selection uses val.jsonl (grouped split, no leakage);
the test set is only used for the final report (--eval_test).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from archa import archclip as A  # noqa: E402
from archa import sketch as S  # noqa: E402
from archa import taxonomy as T  # noqa: E402
from archa.data import read_jsonl  # noqa: E402


class CLIPData(Dataset):
    def __init__(self, jsonl, maps, domain, tokenizer, train, require_known=False, size=224):
        rows = read_jsonl(jsonl)
        if require_known:
            rows = [r for r in rows if r["fields"]["known"]]
        self.rows, self.maps, self.domain, self.tok, self.train = rows, maps, domain, tokenizer, train
        self.tf = A.make_transform(train, domain, size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = random.Random()
        if self.domain == "photo":
            im = Image.open(r["image"]).convert("RGB")
        else:
            im = Image.open(r["cond"]).convert("L")
            if self.train:
                if rng.random() < 0.3:
                    im = S.thicken(im, 3)
                if rng.random() < 0.2:
                    im = S.random_erase_lines(im, rng)
            im = im.convert("RGB")
        single, multi = A.encode_labels(r["fields"], self.maps)
        cap = T.build_caption(r["fields"], drop_prob=0.3 if self.train else 0.0, rng=rng)
        ids = self.tok(cap, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        return {"x": self.tf(im), "ids": ids.input_ids[0], "mask": ids.attention_mask[0],
                "single": torch.tensor([single[h] for h in A.SINGLE]),
                "multi": torch.stack([torch.from_numpy(multi[h]) if multi[h] is not None
                                      else torch.full((len(self.maps[h]),), -1.0) for h in A.MULTI])}


def class_weights(ds, maps):
    w = {}
    for k, h in enumerate(A.SINGLE):
        cnt = np.ones(len(maps[h]))
        for r in ds.rows:
            c = maps[h].get(r["fields"].get(A.SINGLE[h]), -1)
            if c >= 0:
                cnt[c] += 1
        inv = 1.0 / np.sqrt(cnt)
        w[h] = torch.tensor(inv / inv.mean(), dtype=torch.float32)
    return w


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        if tp + fn == 0:
            continue
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, maps):
    model.eval()
    ys = {h: [] for h in A.SINGLE}
    ps = {h: [] for h in A.SINGLE}
    my, mp = [], []
    for b in loader:
        lg = model.logits(model.image_feats(b["x"].to(device)))
        for k, h in enumerate(A.SINGLE):
            ys[h].append(b["single"][:, k].numpy())
            ps[h].append(lg[h].argmax(-1).cpu().numpy())
        for k, h in enumerate(A.MULTI):
            my.append(b["multi"][:, k].numpy())
            mp.append((torch.sigmoid(lg[h]) > 0.5).cpu().numpy())
    res = {}
    for h in A.SINGLE:
        y, p = np.concatenate(ys[h]), np.concatenate(ps[h])
        ok = y >= 0
        res[h] = {"acc": float((y[ok] == p[ok]).mean()), "macro_f1": macro_f1(y[ok], p[ok], len(maps[h])),
                  "n": int(ok.sum())}
    y, p = np.concatenate(my), np.concatenate(mp)
    ok = (y >= 0).all(1)
    y, p = y[ok], p[ok].astype(int)
    f = []
    for c in range(y.shape[1]):
        tp, fp, fn = ((y[:, c] == 1) & (p[:, c] == 1)).sum(), ((y[:, c] == 0) & (p[:, c] == 1)).sum(), \
                     ((y[:, c] == 1) & (p[:, c] == 0)).sum()
        if tp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    res["materials"] = {"macro_f1": float(np.mean(f)) if f else float("nan"), "n": int(ok.sum())}
    res["_mean_macro_f1"] = float(np.nanmean([v["macro_f1"] for v in res.values() if "macro_f1" in v]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="cache")
    ap.add_argument("--domain", default="photo", choices=["photo", "sketch"])
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out_dir", default="runs/archclip_photo")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--contrastive_w", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--require_known", action="store_true", help="train only on known=true rows")
    ap.add_argument("--eval_test", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    tax = json.load(open(os.path.join(a.data_dir, "taxonomy.json")))
    maps = A.label_maps(tax)
    model = A.ArchCLIP(a.backbone, maps, a.lora_r).to(dev)
    tr = CLIPData(os.path.join(a.data_dir, "train.jsonl"), maps, a.domain, model.tokenizer, True, a.require_known, model.size)
    va = CLIPData(os.path.join(a.data_dir, "val.jsonl"), maps, a.domain, model.tokenizer, False, size=model.size)
    L = lambda ds, sh: DataLoader(ds, a.batch_size, shuffle=sh, num_workers=a.workers, drop_last=sh)
    tl, vl = L(tr, True), L(va, False)
    cw = {h: w.to(dev) for h, w in class_weights(tr, maps).items()}

    lora_params = [p for n, p in model.clip.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr},
                             {"params": model.heads.parameters(), "lr": a.head_lr}], weight_decay=0.05)
    total = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=total, pct_start=0.1)
    use_amp = dev == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    best = -1.0
    for ep in range(a.epochs):
        model.train()
        bar = tqdm(tl, desc=f"epoch {ep + 1}/{a.epochs}")
        for b in bar:
            x, ids, mask = b["x"].to(dev), b["ids"].to(dev), b["mask"].to(dev)
            with torch.autocast(dev, dtype=amp_dtype, enabled=use_amp):
                fi = model.image_feats(x)
                ft = model.text_feats(ids, mask)
                lg = model.logits(fi)
            loss = 0.0
            for k, h in enumerate(A.SINGLE):
                y = b["single"][:, k].to(dev)
                if (y >= 0).any():
                    loss = loss + F.cross_entropy(lg[h].float(), y, weight=cw[h], ignore_index=-1,
                                                  label_smoothing=a.label_smoothing)
            for k, h in enumerate(A.MULTI):
                y = b["multi"][:, k].to(dev)
                ok = (y >= 0).all(1)
                if ok.any():
                    loss = loss + F.binary_cross_entropy_with_logits(lg[h].float()[ok], y[ok])
            fi_n, ft_n = F.normalize(fi.float(), dim=-1), F.normalize(ft.float(), dim=-1)
            sim = model.clip.logit_scale.exp().float() * fi_n @ ft_n.T
            tgt = torch.arange(len(sim), device=dev)
            loss = loss + a.contrastive_w * 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            bar.set_postfix(loss=f"{loss.item():.3f}")
        res = evaluate(model, vl, dev, maps)
        print(f"val mean macro-F1 {res['_mean_macro_f1']:.4f} | " +
              " ".join(f"{h}:{res[h]['macro_f1']:.3f}" for h in list(A.SINGLE) + list(A.MULTI)))
        if res["_mean_macro_f1"] > best:
            best = res["_mean_macro_f1"]
            model.save(os.path.join(a.out_dir, "best"), {"domain": a.domain})
            json.dump(res, open(os.path.join(a.out_dir, "val_metrics.json"), "w"), indent=1)
            print("  saved best")

    if a.eval_test:
        best_m = A.ArchCLIP.load(os.path.join(a.out_dir, "best"), dev)
        te = CLIPData(os.path.join(a.data_dir, "test.jsonl"), maps, a.domain, best_m.tokenizer, False, size=best_m.size)
        res = evaluate(best_m, L(te, False), dev, maps)
        json.dump(res, open(os.path.join(a.out_dir, "test_metrics.json"), "w"), indent=1)
        print("TEST:", json.dumps({h: res[h] for h in res}, indent=1))


if __name__ == "__main__":
    main()
