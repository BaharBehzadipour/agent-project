# ArchAgents v2 — ControlNet جدید + ایجنت‌های تصویریِ وظیفه‌محور

هر ایجنت یک **وظیفه** دارد (زمان، مکان، سبک، متریال) و **عملش یک عملیات روی تصویر است** (تولید/ویرایش ماسک‌دار با LoRA مخصوص خودش)،
نه نوشتن پرامپت. هندسه همیشه با ControlNet (که دوباره آموزش می‌دهید) قفل می‌ماند و بعد از هر عمل،
`ArchCLIP` بررسی می‌کند که صفت‌های قبلی خراب نشده باشند (non-regression gate).

```
اسکچ ─► perceive (ArchCLIP-sketch + retrieval) ─► decide: Time → Location → Style → Material
        (consistency prior از داده + VLM اختیاری)
     └► act:  Style(تولید از اسکچ) → Material(ویرایش ناحیهٔ ساختمان) → Location(ویرایش پس‌زمینه)
              هر مرحله چند پارامتر را امتحان می‌کند و بهترینِ «گذرنده از gate» را نگه می‌دارد
```

## ساختار

```
archagents_v2/
  archa/            taxonomy.py (پاک‌سازی/کپشن) · data.py · sketch.py · archclip.py · spec.py (prior سازگاری)
                    generator.py · verifier.py · retrieval.py · llm.py · pipeline.py · agents/
  scripts/          01 … 11  (به ترتیب اجرا می‌شوند، پایین توضیح داده شده)
  requirements.txt
```

## مرحله ۰ — نصب (یک بار)

```bash
cd archagents_v2
pip install -r requirements.txt
# مدل پایه: stable-diffusion-v1-5/stable-diffusion-v1-5 (همان وزن‌های runwayml که قبلاً استفاده کردید؛ مخزن runwayml حذف شده)
```
GPU: برای مرحله ۳ حداقل ~۲۴GB (A100 / L4 / RTX3090) با bf16. روی T4: `--mixed_precision fp16 --grad_checkpointing --batch_size 1 --grad_accum 32 --use_8bit_adam`.
کلید Gemini فقط اگر VLM می‌خواهید: `export GEMINI_API_KEY=...` (هرگز در کد ننویسید؛ کلید OpenRouter که در کامنت‌های `base_llm_agent.py` بود را revoke کنید).

## مرحله ۱ — پاک‌سازی متادیتا (CPU، چند ثانیه)

فایل‌های **اصلی** متادیتا را لازم دارید (نه `metadata.jsonl` ای که `text:""` بازسازی شده): `metadata (2).jsonl` = train (۱۹٬۴۴۰) و `metadata (3).jsonl` = test (۱٬۰۲۳).
`--train_root` / `--test_root` پوشه‌هایی‌اند که زیرشان `images/` و `conditioning_images/` هست.

```bash
python scripts/01_prepare_data.py \
  --train_meta "metadata (2).jsonl" --train_root /home/jovyan/downloads/extracted/dataset/train \
  --test_meta  "metadata (3).jsonl" --test_root  /home/jovyan/downloads/extracted/dataset/test \
  --out_dir clean_data
```
**چک کنید:** خروجی باید «template-echo rows in train: ~1181» (برچسب‌های قالب‌پرامپت که حذف می‌شوند) و چند کپشن نمونه چاپ کند؛ `missing files` باید ۰ باشد.
همچنین «test rows whose title also occurs in train» را می‌گوید (روی متادیتای من ~۳۰۷ از ۱۰۲۳) — یعنی بخشی از test بناهای تکراری train است؛ در مقاله گزارشش کنید یا test را بازتعریف کنید.

## مرحله ۲ — کش تصویر + اسکچ قوی‌تر (CPU، حدود ده‌ها دقیقه برای ۲۰ هزار تصویر)

چرا: اسکچ‌های قبلی ۱ پیکسلی و بسیار کم‌خط بودند و روی تصاویر بزرگ (مثلاً ۲۷۰۰×۳۶۰۰) هنگام کوچک‌کردن به ۵۱۲ تقریباً محو می‌شدند.
`adaptive_canny` آستانه را از گرادیان می‌گیرد، لکه‌ها را حذف می‌کند و خط را ~۳ پیکسل می‌کند.

```bash
python scripts/02_cache_and_sketch.py --in_dir clean_data --out_dir cache --mode adaptive_canny
# اگر می‌خواهید همان اسکچ‌های قبلی (برای مقایسهٔ عادلانه با مقاله) بمانند:  --mode provided
```
**حتماً نگاه کنید:** ۱۰ فایل تصادفی از `cache/sketches/` را کنار `cache/images/` باز کنید. ساختار ساختمان باید قابل تشخیص باشد.

## مرحله ۳ — آموزش دوباره ControlNet

```bash
# تست دودی (چند دقیقه) — مصرف حافظه و سرعت را اینجا می‌بینید، سرعت×۲۰۰۰۰ = زمان کل
accelerate launch scripts/03_train_controlnet.py --data_dir cache --out_dir runs/smoke --max_steps 20 --val_every 10 --save_every 10

# آموزش اصلی
accelerate launch scripts/03_train_controlnet.py --data_dir cache --out_dir runs/cn_v2
# قطع شد؟ همان دستور + --resume
# ادامه از چک‌پوینت قدیمی‌تان:  --controlnet_init "/path/epoch 50000"
```
تفاوت با آموزش قبلی: کپشن ساختاریافته با attribute-dropout (قبلاً `text=""`)، اسکچ قوی + augmentation، ~۱۳ برابر نمونهٔ بیشتر
(batch مؤثر ۳۲، ۲۰k گام)، cosine LR، وزن min-SNR، اعتبارسنجی بصری با اسکچ‌های ثابت در `runs/cn_v2/val_samples/step_*.jpg`.
**هر ۱۰۰۰ گام `val_samples` را نگاه کنید**: ستون‌ها = اسکچ | خروجی | عکس واقعی.

## مرحله ۴ — ارزیابی ControlNet (قدیم در برابر جدید) — تصمیم «ادامه بدهیم یا نه»

```bash
python scripts/04_eval_generation.py --test_jsonl cache/test.jsonl --out_dir eval/cn --models \
   canny_v11=lllyasviel/control_v11p_sd15_canny  old="/path/epoch 50000"  new=runs/cn_v2/last
python scripts/04_eval_generation.py ... --prompt_mode empty --out_dir eval/cn_emptyprompt   # حساسیت به کپشن
```
جدول: `edge_f1/chamfer` (وفاداری به اسکچ)، `ssim/lpips`، `clip_img/clip_txt`، `fid/kid`. اگر `new` از `old` بهتر نبود، قبل از رفتن به مراحل بعد علتش را پیدا کنید (val_samples را ببینید).
نکته: `old` را روی اسکچ‌های ارائه‌شده بسنجید (`--mode provided` در مرحله ۲) یا هر دو را روی یک ورودی.

## مرحله ۵ — ArchCLIP (دو بار)

```bash
python scripts/05_train_archclip.py --data_dir cache --domain photo  --out_dir runs/archclip_photo  --eval_test
python scripts/05_train_archclip.py --data_dir cache --domain sketch --out_dir runs/archclip_sketch --eval_test
```
`photo` = verifier و متریک ارزیابی روی تصویر تولیدی؛ `sketch` = ادراک ایجنت‌ها از اسکچ. انتخاب بهترین مدل با **val** (split گروهی) است.
خروجی `val_metrics.json` / `test_metrics.json` شامل macro-F1 هر head است — این‌ها را در مقاله بیاورید و head ضعیف را به verifier اعتماد نکنید.
(می‌توانید مرحلهٔ ۵ را همزمان با مرحلهٔ ۳ روی GPU دیگر اجرا کنید.)

## مرحله ۶ — ماسک ساختمان/پس‌زمینه

```bash
python scripts/06_make_masks.py --data_dir cache
```
**حتماً نگاه کنید:** `cache/masks_preview/` (ناحیهٔ غیرساختمان تیره شده). اگر ماسک‌ها بد بودند، لیست `BUILDING_WORDS` را در همان فایل عوض کنید.

## مرحله ۷ — سه LoRA وظیفه (روی ControlNet جدیدِ فریز)

```bash
for t in style material context; do
  accelerate launch scripts/07_train_lora.py --task $t --controlnet runs/cn_v2/last --out_dir runs/lora_$t
done
```
style: کپشنِ فقط سبک+قرن، loss روی کل تصویر · material: فقط متریال+رنگ، loss فقط روی ساختمان · context: فقط مکان+محیط، loss فقط روی پس‌زمینه.

## مرحله ۸–۱۱ — ایجنت‌ها

```bash
python scripts/08_build_index.py                                  # ایندکس retrieval (یک بار)
python scripts/10_eval_agents.py --split test --no_llm            # دقت «تصمیم»ها از روی اسکچ (بدون تولید تصویر، سریع)
python scripts/10_eval_agents.py --split test --limit 200         # با VLM (نیاز به GEMINI_API_KEY)
python scripts/09_run_agents.py --sketch my.png --out_dir results/demo                    # یک اسکچ
python scripts/09_run_agents.py --sketch my.png --lock style_category=Gothic continent=Europe materials=stone,glass --out_dir results/g
python scripts/11_eval_leakage.py --n 100 --no_llm                # ماتریس نشت: monolithic در برابر agents
```
`spec_and_trace.json` هر تصمیم، هر کاندیدا و نتیجهٔ gate را ثبت می‌کند (برای تحلیل خطا و ablation).
ablation: `--no_gate`, `--no_consistency`, `--no_llm`.

## چه چیزی را تست کرده‌ام و چه چیزی را نه

روی لپ‌تاپ (بدون GPU و بدون وزن‌های واقعی SD) با **مدل‌های تصادفیِ بسیار کوچک** همهٔ مراحل را از ابتدا تا انتها اجرا کردم و خطاها را رفع کردم:
۱ (روی متادیتای واقعی شما)، ۲ (روی تصاویر نمونهٔ شما)، ۳ (آموزش، ذخیره، resume، اعتبارسنجی)، ۴ (تولید + معیارها؛ LPIPS و FID نیاز به دانلود وزن دارند و در تست دودی رد شدند)، ۵، ۶ (SegFormer واقعی)، ۷ (هر سه task)، ۸، ۹، ۱۰ و ۱۱.

**تست نشده:** سرعت/حافظهٔ واقعی روی GPU، کیفیت نتایج با وزن‌های واقعی، فراخوانی واقعی Gemini، و معنادار بودن اعداد مرحلهٔ ۱۰ و ۱۱ (در تست دودی مدل‌ها تصادفی بودند).
عددهای «پیش‌فرض» (۲۰k گام، lr، rank، strengthها، eps=0.08 در gate) نقطهٔ شروع‌اند، نه مقادیر بهینه‌شده؛ با val_samples و جدول مرحلهٔ ۴ تنظیمشان کنید.
