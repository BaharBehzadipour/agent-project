"""STEP 4 - generate on the test sketches with one or more ControlNets and score them.

Compare the original baseline, your OLD checkpoint and the NEW one in one go:

  python scripts/04_eval_generation.py --test_jsonl cache/test.jsonl --out_dir eval/cn \
      --models canny_v11=lllyasviel/control_v11p_sd15_canny \
               old="/path/epoch 50000" \
               new=runs/cn_v2/last

Metrics (per image, then averaged):
  edge_f1 / chamfer : does the output follow the sketch? (re-extract a sketch from the
                      generated image with the same extractor and compare with the input)
  ssim / psnr / lpips : fidelity to the ground-truth photo
  clip_img          : CLIP-L similarity to the ground-truth photo (semantic fidelity)
  clip_txt          : CLIP-L similarity to the caption
  fid / kid         : distribution realism (clean-fid, if installed)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from archa import sketch as S  # noqa: E402
from archa import taxonomy as T  # noqa: E402
from archa.data import read_jsonl  # noqa: E402


def fit_size(w, h, short=512, max_long=768):
    s = short / min(w, h)
    nw, nh = w * s, h * s
    if max(nw, nh) > max_long:
        k = max_long / max(nw, nh)
        nw, nh = nw * k, nh * k
    return int(round(nw / 8) * 8), int(round(nh / 8) * 8)


def generate(args, name, path, rows):
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(out, r["id"] + ".png"))]
    if not todo:
        return out
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    cn = ControlNetModel.from_pretrained(path, torch_dtype=dt)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model, controlnet=cn, torch_dtype=dt, safety_checker=None).to(dev)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    for r in tqdm(todo, desc=f"generate[{name}]"):
        sk = Image.open(r["cond"]).convert("RGB")
        size = fit_size(*sk.size, short=args.short, max_long=int(args.short * 1.5))
        sk = sk.resize(size, Image.BOX)
        prompt = "" if args.prompt_mode == "empty" else r["caption"]
        g = torch.Generator(device=dev).manual_seed(args.seed)
        img = pipe(prompt, image=sk, width=size[0], height=size[1], num_inference_steps=args.steps,
                   guidance_scale=args.guidance, generator=g).images[0]
        img.save(os.path.join(out, r["id"] + ".png"))
    del pipe, cn
    torch.cuda.empty_cache()
    return out


def edge_scores(gen_img, sketch_img, tol=3):
    from scipy.ndimage import distance_transform_edt as edt
    gen_sk = np.array(S.make_sketch(gen_img).convert("L")) < 128
    in_sk = np.array(sketch_img.convert("L").resize(gen_img.size, Image.BOX)) < 128
    if in_sk.sum() == 0 or gen_sk.sum() == 0:
        return 0.0, float("nan")
    d_in, d_gen = edt(~in_sk), edt(~gen_sk)
    prec = (d_in[gen_sk] <= tol).mean()       # generated edges that lie on the sketch
    rec = (d_gen[in_sk] <= tol).mean()        # sketch lines that were reproduced
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    chamfer = 0.5 * (d_in[gen_sk].mean() + d_gen[in_sk].mean())
    return float(f1), float(chamfer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", default="cache/test.jsonl")
    ap.add_argument("--out_dir", default="eval/cn")
    ap.add_argument("--models", nargs="+", required=True, help="name=path_or_hf_id ...")
    ap.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--prompt_mode", default="full", choices=["full", "empty"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only_clean", action="store_true", help="skip rows with template-echo labels")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--short", type=int, default=512, help="short side of generated images")
    ap.add_argument("--skip_generation", action="store_true")
    ap.add_argument("--no_lpips", action="store_true", help="skip LPIPS (needs to download AlexNet weights)")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_jsonl(args.test_jsonl)
    if args.only_clean:
        rows = [r for r in rows if not r.get("flag_placeholder")]
    if args.limit:
        rows = rows[:args.limit]
    models = dict(m.split("=", 1) for m in args.models)

    import lpips
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = None if args.no_lpips else lpips.LPIPS(net="alex").to(dev)
    clip = CLIPModel.from_pretrained(args.clip_model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_model)

    @torch.no_grad()
    def clip_feats(imgs=None, texts=None):
        if imgs is not None:
            x = proc(images=imgs, return_tensors="pt").to(dev)
            f = clip.get_image_features(**x)
        else:
            x = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
            f = clip.get_text_features(**x)
        return torch.nn.functional.normalize(f, dim=-1)

    gt_dir = os.path.join(args.out_dir, "_gt")
    os.makedirs(gt_dir, exist_ok=True)
    summary = {}
    for name, path in models.items():
        gen_dir = os.path.join(args.out_dir, name) if args.skip_generation else generate(args, name, path, rows)
        per = []
        for r in tqdm(rows, desc=f"score[{name}]"):
            gp = os.path.join(gen_dir, r["id"] + ".png")
            if not os.path.exists(gp):
                continue
            gen = Image.open(gp).convert("RGB")
            gt = Image.open(r["image"]).convert("RGB").resize(gen.size, Image.BICUBIC)
            gtp = os.path.join(gt_dir, r["id"] + ".png")
            if not os.path.exists(gtp):
                gt.save(gtp)
            f1, ch = edge_scores(gen, Image.open(r["cond"]))
            a, b = np.array(gen), np.array(gt)
            t = lambda im: torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
            with torch.no_grad():
                lpv = lp(t(gen), t(gt)).item() if lp is not None else float("nan")
            ci = (clip_feats([gen]) @ clip_feats([gt]).T).item()
            ct = (clip_feats([gen]) @ clip_feats(texts=[r["caption"]]).T).item()
            per.append({"id": r["id"], "edge_f1": f1, "chamfer": ch,
                        "ssim": ssim(a, b, channel_axis=2, data_range=255),
                        "psnr": psnr(b, a, data_range=255), "lpips": lpv,
                        "clip_img": ci, "clip_txt": ct})
        keys = [k for k in per[0] if k != "id"]
        summary[name] = {k: float(np.nanmean([p[k] for p in per])) for k in keys}
        summary[name]["n"] = len(per)
        json.dump(per, open(os.path.join(args.out_dir, f"per_image_{name}.json"), "w"))
        try:
            from cleanfid import fid
            summary[name]["fid"] = float(fid.compute_fid(gen_dir, gt_dir, verbose=False))
            summary[name]["kid"] = float(fid.compute_kid(gen_dir, gt_dir, verbose=False))
        except Exception as e:  # clean-fid missing or too few images
            print("FID/KID skipped:", e)

    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    cols = ["edge_f1", "chamfer", "ssim", "lpips", "clip_img", "clip_txt", "fid", "kid", "n"]
    print("\n| model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for name, s in summary.items():
        print(f"| {name} | " + " | ".join(f"{s[c]:.3f}" if c in s and c != "n" else str(s.get(c, '-'))
                                          for c in cols) + " |")


if __name__ == "__main__":
    main()
