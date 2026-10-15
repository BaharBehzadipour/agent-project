"""STEP 3 - (re)train the ControlNet on SD-1.5.

What is different from the first run (and why):
  * structured captions with attribute-dropout + 10% empty caption  (old run used text="")
  * stronger sketches (step 2) + sketch augmentation (thickness, missing lines)
  * ~13x more samples seen (effective batch 32, 20k steps), cosine LR, min-SNR loss weight
  * init from the Canny ControlNet v1.1 (or from your old checkpoint)
  * periodic validation grids with FIXED sketches, auto-resume, rolling checkpoints

Run:
  accelerate launch scripts/03_train_controlnet.py --data_dir cache --out_dir runs/cn_v2
Smoke test (a few steps, any GPU):
  accelerate launch scripts/03_train_controlnet.py --data_dir cache --out_dir runs/smoke --max_steps 20 \
         --val_every 10 --save_every 10
"""
import argparse
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from diffusers import (AutoencoderKL, ControlNetModel, DDPMScheduler,  # noqa: E402
                       StableDiffusionControlNetPipeline, UNet2DConditionModel,
                       UniPCMultistepScheduler)
from diffusers.optimization import get_scheduler  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

from archa import taxonomy as T  # noqa: E402
from archa.data import ControlDataset, collate, read_jsonl  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="cache", help="folder with train.jsonl / val.jsonl from step 2")
    ap.add_argument("--out_dir", default="runs/cn_v2")
    ap.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--controlnet_init", default="lllyasviel/control_v11p_sd15_canny",
                    help="HF id or local folder (e.g. your old 'epoch 50000' checkpoint)")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=20000, help="optimizer steps")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--attr_drop", type=float, default=0.25)
    ap.add_argument("--empty_prob", type=float, default=0.10)
    ap.add_argument("--min_snr_gamma", type=float, default=5.0, help="0 disables")
    ap.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"],
                    help="bf16 on A100/H100/L4, fp16 on T4/V100")
    ap.add_argument("--use_8bit_adam", action="store_true")
    ap.add_argument("--grad_checkpointing", action="store_true", help="saves memory, ~25% slower")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out_dir>/last")
    return ap.parse_args()


def make_val_batch(args, tokenizer, n=4):
    """n fixed validation sketches (one per different style) + their full captions."""
    rows = read_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    seen, pick = set(), []
    for r in rows:
        sc = r["fields"]["style_category"]
        if sc not in seen and r["fields"]["century"] is not None:
            seen.add(sc)
            pick.append(r)
        if len(pick) == n:
            break
    return pick


def validate(args, accelerator, controlnet, vae, text_encoder, tokenizer, unet, step, weight_dtype, rows):
    pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=accelerator.unwrap_model(controlnet),
        scheduler=UniPCMultistepScheduler.from_pretrained(args.base_model, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    ).to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)
    tiles = []
    for r in rows:
        sk = Image.open(r["cond"]).convert("RGB")
        w, h = sk.size
        s = args.resolution / min(w, h)
        sk = sk.resize((round(w * s) // 8 * 8, round(h * s) // 8 * 8), Image.BOX)
        sk = sk.crop(((sk.width - args.resolution) // 2, (sk.height - args.resolution) // 2,
                      (sk.width - args.resolution) // 2 + args.resolution,
                      (sk.height - args.resolution) // 2 + args.resolution))
        prompt = T.build_caption(r["fields"])
        g = torch.Generator(device=accelerator.device).manual_seed(1234)
        with torch.autocast(accelerator.device.type, dtype=weight_dtype, enabled=weight_dtype != torch.float32):
            out = pipe(prompt, image=sk, num_inference_steps=25, generator=g, guidance_scale=7.5).images[0]
        gt = Image.open(r["image"]).convert("RGB").resize(sk.size, Image.BICUBIC)
        tiles.append([sk, out, gt])
    W = args.resolution
    grid = Image.new("RGB", (W * 3, W * len(tiles)), "white")
    for i, row in enumerate(tiles):
        for j, im in enumerate(row):
            grid.paste(im.resize((W, W)), (j * W, i * W))
    os.makedirs(os.path.join(args.out_dir, "val_samples"), exist_ok=True)
    grid.save(os.path.join(args.out_dir, "val_samples", f"step_{step:06d}.jpg"), quality=90)
    with open(os.path.join(args.out_dir, "val_samples", f"step_{step:06d}.txt"), "w") as f:
        f.write("\n".join(T.build_caption(r["fields"]) for r in rows))
    del pipe
    torch.cuda.empty_cache()


def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision=args.mixed_precision)
    set_seed(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    start_step = 0
    init = args.controlnet_init
    last = os.path.join(args.out_dir, "last")
    if args.resume and os.path.isdir(last):
        init = last
        start_step = json.load(open(os.path.join(last, "state.json")))["step"]
        print(f"resuming from {last} at step {start_step}")
    controlnet = ControlNetModel.from_pretrained(init)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    if args.grad_checkpointing:
        controlnet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
    else:
        opt_cls = torch.optim.AdamW
    optimizer = opt_cls(controlnet.parameters(), lr=args.lr, weight_decay=1e-2)

    train_ds = ControlDataset(os.path.join(args.data_dir, "train.jsonl"), tokenizer, args.resolution,
                              train=True, attr_drop=args.attr_drop, empty_prob=args.empty_prob)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=collate, num_workers=args.workers,
                                         drop_last=True, persistent_workers=args.workers > 0)
    lr_sched = get_scheduler("cosine", optimizer=optimizer,
                             num_warmup_steps=args.warmup * accelerator.num_processes,
                             num_training_steps=args.max_steps * accelerator.num_processes)
    for _ in range(start_step * accelerator.num_processes):
        lr_sched.step()
    controlnet, optimizer, loader, lr_sched = accelerator.prepare(controlnet, optimizer, loader, lr_sched)

    val_rows = make_val_batch(args, tokenizer)
    snr_cache = None
    if args.min_snr_gamma > 0:
        ac = noise_scheduler.alphas_cumprod.to(accelerator.device)
        snr_cache = ac / (1 - ac)

    eff = args.batch_size * args.grad_accum * accelerator.num_processes
    accelerator.print(f"train samples {len(train_ds)} | effective batch {eff} | "
                      f"epochs ~ {args.max_steps * eff / len(train_ds):.1f}")
    json.dump(vars(args), open(os.path.join(args.out_dir, "args.json"), "w"), indent=1)

    step = start_step
    bar = tqdm(total=args.max_steps, initial=step, disable=not accelerator.is_local_main_process)
    running = 0.0
    while step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(controlnet):
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    enc = text_encoder(batch["input_ids"])[0]
                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=latents.device).long()
                noisy = noise_scheduler.add_noise(latents, noise, t)

                down, mid = controlnet(noisy, t, encoder_hidden_states=enc,
                                       controlnet_cond=batch["conditioning_pixel_values"].to(weight_dtype),
                                       return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=enc,
                            down_block_additional_residuals=[d.to(weight_dtype) for d in down],
                            mid_block_additional_residual=mid.to(weight_dtype)).sample

                loss = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=[1, 2, 3])
                if snr_cache is not None:
                    snr = snr_cache[t]
                    w = torch.clamp(snr, max=args.min_snr_gamma) / snr
                    loss = loss * w
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.detach().item() / args.grad_accum
            if accelerator.sync_gradients:
                step += 1
                bar.update(1)
                if step % 20 == 0:
                    bar.set_postfix(loss=f"{running / 20:.4f}", lr=f"{lr_sched.get_last_lr()[0]:.2e}")
                    running = 0.0
                if accelerator.is_main_process:
                    if step % args.save_every == 0 or step == args.max_steps:
                        save(accelerator, controlnet, args, step)
                    if step % args.val_every == 0 or step == 1:
                        controlnet.eval()
                        validate(args, accelerator, controlnet, vae, text_encoder, tokenizer, unet,
                                 step, weight_dtype, val_rows)
                        controlnet.train()
                if step >= args.max_steps:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save(accelerator, controlnet, args, step)
    print("done ->", os.path.join(args.out_dir, "last"))


def save(accelerator, controlnet, args, step):
    ck = os.path.join(args.out_dir, f"ckpt-{step:06d}")
    accelerator.unwrap_model(controlnet).save_pretrained(ck)
    json.dump({"step": step}, open(os.path.join(ck, "state.json"), "w"))
    last = os.path.join(args.out_dir, "last")
    if os.path.isdir(last):
        shutil.rmtree(last)
    shutil.copytree(ck, last)
    cks = sorted(d for d in os.listdir(args.out_dir) if d.startswith("ckpt-"))
    for d in cks[:-args.keep]:
        shutil.rmtree(os.path.join(args.out_dir, d), ignore_errors=True)


if __name__ == "__main__":
    main()
