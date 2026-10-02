# Medsynth: train once with `python train.py`, use the model forever

One command fine-tunes Stable Diffusion 1.5 (LoRA, on your NVIDIA GPU) on three public
skin-image datasets and saves the result **inside this project folder**. `main.py` loads only
that saved model, so generating an image never retrains anything.

```bash
python train.py
```

## The three datasets = three imaging views

One model learns all three views. Each view has its own caption prefix, so you can ask for
the one you want.

| View | Dataset | Caption prefix | Example caption |
|---|---|---|---|
| Dermoscopy | **HAM10000** (~10k images, 7 diagnoses) | `dermoscopy image of` | `dermoscopy image of melanoma, malignant, back` |
| Smartphone close-up | **PAD-UFES-20** (~2.3k images, 7 diagnoses) | `smartphone photo of` | `smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 2` |
| Clinical photo | **Fitzpatrick17k** (~16k images, 114 conditions) | `clinical photo of` | `clinical photo of psoriasis, non-neoplastic, Fitzpatrick skin type 5` |

HAM10000 is the curated subset of the ISIC archive. The full ISIC archive is much larger and
needs its own downloader.

---

## 1. Project layout

```
medsynth/
├── main.py
├── train.py                  <- everything is configured at the top of this file
├── requirements.txt
├── utils/chat_history.py
├── data/                     <- your datasets
│   └── raw/ham10000  raw/pad_ufes_20  raw/fitzpatrick17k
└── medsynth-model/           <- created by train.py: your trained model
    ├── pipeline/             <- what main.py loads (standalone, ~2 GB)
    ├── lora/  checkpoints/  state/  samples/
    └── model_info.json  train_config.json  train_log.csv  prompt_examples.txt
```

## 2. Install

You need an **NVIDIA GPU with CUDA** (8 GB VRAM minimum, 16 GB+ comfortable), Python 3.10 to
3.12, and about 30 GB of free disk.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

# 1) CUDA build of PyTorch: get the exact command for your driver from
#    https://pytorch.org/get-started/locally   (example only:)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) everything else
pip install -r requirements.txt

# 3) the GPU must be visible; this must print True and your GPU name
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`train.py` repeats this check on start and **refuses to run on CPU**, explaining how to fix a
CPU-only PyTorch install. The 5 GB base model downloads from Hugging Face on the first run
(no token needed). The old `runwayml/stable-diffusion-v1-5` repo was removed, so the default is
its community mirror `stable-diffusion-v1-5/stable-diffusion-v1-5`.

## 3. Get the datasets

| Dataset | How | Where it ends up |
|---|---|---|
| **Fitzpatrick17k** | **Automatic.** `train.py` downloads the CSV and images itself. | `data/raw/fitzpatrick17k/` |
| **HAM10000** | Manual. https://doi.org/10.7910/DVN/DBW86T (or Kaggle `kmader/skin-cancer-mnist-ham10000`). Get `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`, `HAM10000_metadata`. Licence CC BY-NC 4.0. | `data/raw/ham10000/` |
| **PAD-UFES-20** | Manual. https://data.mendeley.com/datasets/zr7vgbcyr2/1 (or Kaggle `mahdavi1202/skin-cancer`). Get the image zip(s) and `metadata.csv`. | `data/raw/pad_ufes_20/` |

- **Loose files are fine.** If you drop the HAM10000 or PAD-UFES-20 files straight into
  `data/`, `train.py` moves them into the right `raw/` folder for you. Zips stay zipped
  (nested zips too); the script extracts them.
- **Fitzpatrick17k notes.** About 16.4k usable rows remain after dropping images the authors
  flagged as wrong or unusable. Some source links are dead; those are recorded in
  `download_failures.tsv` and not retried on later runs. The first download takes a while, and
  if you stop it, the next run continues where it left off. The images belong to third-party
  sites, so see the dataset's GitHub page for terms.
- If a dataset is missing, `train.py` stops before doing any slow work and prints exactly what
  to download and where.

## 4. Train: one command

```bash
python train.py
```

It runs these steps in order and skips whatever is already done:

1. checks for CUDA (fails fast if missing)
2. tidies loose dataset files into `data/raw/<dataset>/`
3. downloads the Fitzpatrick17k images
4. builds captions, caps large classes, resizes everything to 512×512, holds out 2 % for evaluation
5. trains the LoRA on the GPU
6. merges it into a standalone model at `medsynth-model/pipeline/` and marks the run complete

**Stopped or crashed?** Just run `python train.py` again. It resumes from the newest
checkpoint (saved every 1,000 steps). **Already finished?** Running it again prints where the
model is and exits, so you can never retrain by accident.

**First-time check:** set `smoke_test = True` in the config (below) and run it once. It uses a
small slice of each dataset, 40 steps, and writes to its own `medsynth-smoketest/` folder.
Then set it back to `False`.

### What one epoch looks like
Every epoch draws from the three views in fixed proportions (35 % dermoscopy, 25 % smartphone,
40 % clinical) so the small PAD-UFES-20 set is not drowned out. Each view also gets only the
augmentations that make sense for it: dermoscopy is flipped and rotated freely; smartphone and
clinical photos are only mirrored, so gravity and anatomy stay realistic.

### GPU speed
Training runs on CUDA with mixed precision (bf16 or fp16), TF32, cuDNN autotuning and a fused
optimizer. The batch settings are chosen from your VRAM automatically while keeping 16 images
per optimizer step. These thresholds are estimates:

| VRAM | Batch × accumulation | Gradient checkpointing |
|---|---|---|
| 20 GB+ | 4 × 4 | off |
| 14–20 GB | 2 × 8 | off |
| 10–14 GB | 2 × 8 | on |
| under 10 GB | 1 × 16 | on |

Rough time for the default 15,000 steps is on the order of half a day on a 24 GB card and
longer on slower ones. Check the `it/s` in the progress bar, and look at `medsynth-model/samples/`
(4 preview images, one per view, every 1,000 steps).

## 5. Configuration (the only place you edit)

There are no command-line arguments. Open `train.py` and edit the **CONFIG** section near the top.

### Per dataset (`DATASETS`)

| Setting | HAM10000 | PAD-UFES-20 | Fitzpatrick17k | Meaning |
|---|---|---|---|---|
| `caption_prefix` | dermoscopy image of | smartphone photo of | clinical photo of | how the view is named in prompts |
| `sample_share` | 0.35 | 0.25 | 0.40 | share of each epoch |
| `max_per_class` | 1200 | 1200 | 600 | cap per diagnosis so common ones don't dominate |
| `min_side` | 256 | 128 | 128 | drop images smaller than this (avoids blurry upscaling) |
| `hflip / vflip / rot90` | yes / yes / yes | yes / no / no | yes / no / no | safe augmentations |
| `use_category` | yes | yes | yes | add benign / malignant / pre-malignant |
| `use_body_site` | yes | yes | no (not in data) | add e.g. "back", "face" |
| `use_skin_type` | no (not in data) | yes | yes | add "Fitzpatrick skin type N" |

### General (`Config`)

| Setting | Default | Notes |
|---|---|---|
| `max_train_steps` | 15000 | try 5000 for a quicker first model |
| `effective_batch` / `auto_batch` | 16 / True | set `auto_batch = False` to use `batch_size`, `grad_accum`, `gradient_checkpointing` yourself |
| `learning_rate`, `rank` | 1e-4, 32 | LoRA rank 64 gives more capacity |
| `resolution` | 512 | |
| `require_all_datasets` | True | set False to train on whichever datasets you have |
| `retrain` | False | True discards checkpoints and trains from scratch (set it back afterwards) |
| `smoke_test` | False | quick end-to-end check |
| `num_workers` | 2 (0 on Windows) | |

## 6. Run the app

```bash
python main.py
```

`main.py` loads `medsynth-model/pipeline` from disk (offline) onto the GPU once at startup and
reuses it for every prompt. To use a model stored elsewhere:
`MEDSYNTH_MODEL_DIR=/path/to/pipeline python main.py`.

### Writing prompts
Prompts work best in the same grammar as the training captions. Choose **Dermoscopy /
Smartphone / Clinical photo** in the app; with *Clinical prompt enhancement* on, the matching
prefix is added for you, so you type the rest:

```
melanoma, malignant, back
basal cell carcinoma, malignant, face, Fitzpatrick skin type 2
psoriasis, non-neoplastic, Fitzpatrick skin type 5
```

`medsynth-model/prompt_examples.txt` lists one example caption per class (HAM10000 has 7,
PAD-UFES-20 has 7, Fitzpatrick17k has 114). Use the skin-type phrase (1 to 6) to steer skin
tone. Good settings: 30 steps, guidance 6 to 8.

### Using the model in other projects

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "medsynth-model/pipeline", torch_dtype=torch.float16, local_files_only=True
).to("cuda")
image = pipe("dermoscopy image of melanoma, malignant, back",
             num_inference_steps=30, guidance_scale=6.5).images[0]
```
Back up `medsynth-model/pipeline`. For a small file to share, use
`medsynth-model/lora/pytorch_lora_weights.safetensors` with `pipe.load_lora_weights(...)` on the
same SD 1.5 base.

## 7. Check quality
`data/prepared/heldout.jsonl` lists 2 % of images kept out of training. Generate from their
captions and compare with those real images (FID/KID), or train a classifier on synthetic images
and test it on real ones. If the previews stop improving, or start copying training images, use
an earlier `medsynth-model/checkpoints/lora-step-*`.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA is not available ... CPU-only build` | Reinstall PyTorch with CUDA (section 2). |
| `CUDA out of memory` | Set `auto_batch = False`, `batch_size = 1`, `grad_accum = 16`, `gradient_checkpointing = True`. |
| Loss becomes `nan` with fp16 | Set `mixed_precision = "bf16"` (RTX 30-series and newer) or `learning_rate = 5e-5`. |
| Cannot start: some datasets are not ready | Follow the printed instructions; folder names must be `ham10000`, `pad_ufes_20`, `fitzpatrick17k`. |
| App says "Trained model not found" | Training hasn't finished, or the model isn't at `medsynth-model/pipeline`. |
| Many Fitzpatrick downloads fail | Expected for dead links. Set `retry_failed_downloads = True` to try them again. |

## 9. Read this before using the results

- The output is **synthetic data**. It must not be used for diagnosis or presented as real
  patient images.
- HAM10000 (CC BY-NC) and the Fitzpatrick17k image sources restrict commercial use. Treat the
  trained weights as **research/non-commercial** unless you have cleared every licence.
- The datasets are skewed (HAM10000 toward lighter skin, PAD-UFES-20 from one Brazilian region).
  Fitzpatrick17k plus the skin-type captions helps, but check outputs across skin tones.
- The benign / malignant / pre-malignant words are a simplified mapping made for prompting, not
  clinical ground truth. Have a dermatologist review samples if the images feed research.