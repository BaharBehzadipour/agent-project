"""STEP 7 - train a task LoRA (style | material | context) on top of the FROZEN new ControlNet.

Each LoRA only sees ITS caption fields, and (for material / context) its loss is computed only
inside its own region, so it learns to act only there:
    style    : caption = style + century              loss on the whole image
    material : caption = materials + color            loss on the BUILDING mask
    context  : caption = location + environment       loss on the BACKGROUND (non-building) mask

Run (3 times):
  accelerate launch scripts/07_train_lora.py --task style    --controlnet runs/cn_v2/last --out_dir runs/lora_style
  accelerate launch scripts/07_train_lora.py --task material --controlnet runs/cn_v2/last --out_dir runs/lora_material
  accelerate launch scripts/07_train_lora.py --task context  --controlnet runs/cn_v2/last --out_dir runs/lora_context
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from diffusers import (AutoencoderKL, ControlNetModel, DDPMScheduler,  # noqa: E402
                       StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.optimization import get_scheduler  # noqa: E402
from diffusers.utils import convert_state_dict_to_diffusers  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

from archa.data import ControlDataset, collate  # noqa: E402

TASKS = {
    "style":    {"only": ["style", "century"],         "mask": None},
    "material": {"only": ["materials", "color"],       "mask": "building"},
    "context":  {"only": ["location", "environment"],  "mask": "background"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--data_dir", default="cache")
    ap.add_argument("--controlnet", required=True, help="folder of the ControlNet trained in step 3")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--empty_prob", type=float, default=0.1)
    ap.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = TASKS[a.task]
    os.makedirs(a.out_dir, exist_ok=True)
    acc = Accelerator(gradient_accumulation_steps=a.grad_accum, mixed_precision=a.mixed_precision)
    set_seed(a.seed)

    tok = CLIPTokenizer.from_pretrained(a.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(a.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(a.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(a.base_model, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(a.controlnet)
    sched = DDPMScheduler.from_pretrained(a.base_model, subfolder="scheduler")
    for m in (vae, unet, text_encoder, controlnet):
        m.requires_grad_(False)
    wd = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.mixed_precision]
    for m in (vae, text_encoder, controlnet, unet):
        m.to(acc.device, dtype=wd)
    unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:                       # LoRA weights in fp32 for stable training
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)

    masks_dir = os.path.join(a.data_dir, "masks") if cfg["mask"] else None
    ds = ControlDataset(os.path.join(a.data_dir, "train.jsonl"), tok, a.resolution, train=True,
                        attr_drop=0.0, empty_prob=a.empty_prob, only_fields=cfg["only"],
                        masks_dir=masks_dir, mask_region=cfg["mask"] or "building", sketch_aug=True)
    loader = torch.utils.data.DataLoader(ds, a.batch_size, shuffle=True, collate_fn=collate,
                                         num_workers=a.workers, drop_last=True)
    lr_s = get_scheduler("cosine", optimizer=opt, num_warmup_steps=100 * acc.num_processes,
                         num_training_steps=a.max_steps * acc.num_processes)
    unet, opt, loader, lr_s = acc.prepare(unet, opt, loader, lr_s)
    acc.print(f"task={a.task} fields={cfg['only']} mask={cfg['mask']} samples={len(ds)}")
    json.dump({**vars(a), **cfg}, open(os.path.join(a.out_dir, "args.json"), "w"), indent=1)

    step, bar = 0, tqdm(total=a.max_steps, disable=not acc.is_local_main_process)
    while step < a.max_steps:
        for b in loader:
            with acc.accumulate(unet):
                with torch.no_grad():
                    lat = vae.encode(b["pixel_values"].to(wd)).latent_dist.sample() * vae.config.scaling_factor
                    enc = text_encoder(b["input_ids"])[0]
                noise = torch.randn_like(lat)
                t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=lat.device).long()
                noisy = sched.add_noise(lat, noise, t)
                with torch.no_grad():
                    down, mid = controlnet(noisy, t, encoder_hidden_states=enc,
                                           controlnet_cond=b["conditioning_pixel_values"].to(wd), return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=enc,
                            down_block_additional_residuals=[d.to(wd) for d in down],
                            mid_block_additional_residual=mid.to(wd)).sample
                mse = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(1, keepdim=True)
                if "loss_mask" in b:
                    m = F.interpolate(b["loss_mask"][:, None].to(mse.device), size=mse.shape[-2:], mode="area")
                    loss = (mse * m).sum() / m.sum().clamp(min=1.0)
                else:
                    loss = mse.mean()
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(params, 1.0)
                opt.step()
                lr_s.step()
                opt.zero_grad(set_to_none=True)
            if acc.sync_gradients:
                step += 1
                bar.update(1)
                bar.set_postfix(loss=f"{loss.item():.4f}")
                if acc.is_main_process and (step % a.save_every == 0 or step == a.max_steps):
                    save(acc, unet, a.out_dir)
                if step >= a.max_steps:
                    break
    acc.wait_for_everyone()
    print("done ->", a.out_dir)


def save(acc, unet, out_dir):
    u = acc.unwrap_model(unet)
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(u))
    StableDiffusionPipeline.save_lora_weights(out_dir, unet_lora_layers=sd, safe_serialization=True)


if __name__ == "__main__":
    main()
