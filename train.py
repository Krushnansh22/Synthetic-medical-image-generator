#!/usr/bin/env python3
"""
Medsynth trainer.  Run it with one command, no arguments:

    python train.py

It fine-tunes Stable Diffusion 1.5 (LoRA, on the NVIDIA GPU) on three datasets
that together cover three imaging views, then saves a standalone model to
<project>/medsynth-model/pipeline, which app.py uses.

    view              dataset          caption prefix
    ----------------  ---------------  ----------------------
    dermoscopy        HAM10000         "dermoscopy image of"
    smartphone close  PAD-UFES-20      "smartphone photo of"
    clinical photo    Fitzpatrick17k   "clinical photo of"

What one run does, in order (every step is skipped when already done):
    1. checks that CUDA is available
    2. tidies loose dataset files into data/raw/<dataset>/
    3. downloads the Fitzpatrick17k images (HAM10000 / PAD-UFES-20 are manual)
    4. builds captions, balances classes, resizes to 512x512
    5. trains (resumes automatically after a crash or Ctrl+C)
    6. merges the LoRA into a standalone pipeline and marks the model complete

Re-running `python train.py` once training is complete does nothing except tell
you the model is ready, so you never retrain by accident.

All settings live in the CONFIG section below. Edit them there.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:  # torch is only needed for the training part
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


# ##########################################################################
#                              C O N F I G
# ##########################################################################
PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    """Everything that is specific to one dataset / imaging view."""

    name: str
    folder: str            # sub-folder of data/raw
    view: str              # imaging view this dataset teaches the model
    caption_prefix: str    # start of every caption, e.g. "dermoscopy image of"
    sample_share: float    # share of each training epoch drawn from this dataset
    max_per_class: int     # cap per diagnosis so common classes don't dominate (0 = none)
    min_side: int          # drop images whose shorter side is below this many pixels
    hflip: bool            # augmentations that are anatomically safe for this view
    vflip: bool
    rot90: bool
    use_category: bool     # add "benign" / "malignant" / ... to the caption
    use_body_site: bool    # add the body site to the caption (if the dataset has it)
    use_skin_type: bool    # add "Fitzpatrick skin type N" to the caption
    # code -> (diagnosis name, category); used by HAM10000 and PAD-UFES-20
    label_map: dict = field(default_factory=dict)


# The category words (benign / malignant / pre-malignant) are a simplified
# mapping made for prompting convenience, not diagnostic ground truth.
DATASETS: dict[str, DatasetSpec] = {
    "ham10000": DatasetSpec(
        name="ham10000", folder="ham10000", view="dermoscopy",
        caption_prefix="dermoscopy image of",
        sample_share=0.35, max_per_class=1200, min_side=256,
        hflip=True, vflip=True, rot90=True,          # dermoscopy has no "up"
        use_category=True, use_body_site=True, use_skin_type=False,
        label_map={
            "akiec": ("actinic keratosis or intraepithelial carcinoma", "pre-malignant"),
            "bcc": ("basal cell carcinoma", "malignant"),
            "bkl": ("benign keratosis-like lesion", "benign"),
            "df": ("dermatofibroma", "benign"),
            "mel": ("melanoma", "malignant"),
            "nv": ("melanocytic nevus", "benign"),
            "vasc": ("vascular lesion", "benign"),
        },
    ),
    "pad_ufes_20": DatasetSpec(
        name="pad_ufes_20", folder="pad_ufes_20", view="smartphone close-up",
        caption_prefix="smartphone photo of",
        sample_share=0.25, max_per_class=1200, min_side=128,
        hflip=True, vflip=False, rot90=False,        # real photos: keep gravity
        use_category=True, use_body_site=True, use_skin_type=True,
        label_map={
            "BCC": ("basal cell carcinoma", "malignant"),
            "SCC": ("squamous cell carcinoma", "malignant"),
            "ACK": ("actinic keratosis", "pre-malignant"),
            "SEK": ("seborrheic keratosis", "benign"),
            "BOD": ("Bowen's disease", "malignant"),  # squamous cell carcinoma in situ
            "MEL": ("melanoma", "malignant"),
            "NEV": ("melanocytic nevus", "benign"),
        },
    ),
    "fitzpatrick17k": DatasetSpec(
        name="fitzpatrick17k", folder="fitzpatrick17k", view="clinical photo",
        caption_prefix="clinical photo of",
        sample_share=0.40, max_per_class=600, min_side=128,
        hflip=True, vflip=False, rot90=False,
        use_category=True, use_body_site=False, use_skin_type=True,
    ),
}

FITZ_CSV_URL = (
    "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
)
# Fitzpatrick17k rows whose manual QC flag starts with these digits are
# "3 Wrongly labelled", "4 Other" or "5 Potentially ..." and are dropped.
FITZ_BAD_QC = ("3", "4", "5")

MANUAL_HELP = {
    "ham10000": (
        "Download HAM10000_images_part_1.zip, HAM10000_images_part_2.zip and "
        "HAM10000_metadata from https://doi.org/10.7910/DVN/DBW86T (or Kaggle: "
        "kmader/skin-cancer-mnist-ham10000) and put them in {folder}"
    ),
    "pad_ufes_20": (
        "Download the images and metadata.csv from "
        "https://data.mendeley.com/datasets/zr7vgbcyr2/1 and put them in {folder}"
    ),
    "fitzpatrick17k": (
        "The images are downloaded automatically. Check your internet connection "
        "and re-run, or see README.md section 3."
    ),
}

# Loose files that get moved into data/raw/<dataset>/ if you dropped them
# straight into data/ (e.g. HAM10000 files next to the raw/ folder).
ADOPT_PATTERNS = {
    "ham10000": ["HAM10000_metadata*", "HAM10000_images_part_*"],
    "pad_ufes_20": ["PAD-UFES-20*", "imgs_part_*"],
}

DEFAULT_VALIDATION_PROMPTS = [
    "dermoscopy image of melanoma, malignant, back",
    "dermoscopy image of melanocytic nevus, benign, upper extremity",
    "smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 3",
    "clinical photo of psoriasis, non-neoplastic, Fitzpatrick skin type 5",
]


@dataclass
class Config:
    # ---- where things live (all inside the project folder) ----------------
    data_root: Path = PROJECT_DIR / "data"
    output_dir: Path = PROJECT_DIR / "medsynth-model"   # trained model is stored here
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    # ---- behaviour --------------------------------------------------------
    require_all_datasets: bool = True     # stop with instructions if one is missing
    auto_download_fitzpatrick: bool = True
    retry_failed_downloads: bool = False  # retry links that failed before
    fitz_workers: int = 6
    retrain: bool = False                 # True = discard checkpoints, train from scratch
    smoke_test: bool = False              # True = ~10 minute end-to-end check
    allow_cpu: bool = False               # only for tests; real training needs CUDA

    # ---- data -------------------------------------------------------------
    resolution: int = 512
    holdout: float = 0.02                 # kept out of training for evaluation
    min_images_per_dataset: int = 50
    limit_per_dataset: int = 0            # debug: keep only N images per dataset
    fitz_limit: int = 0                   # debug: download only N Fitzpatrick images

    # ---- training ---------------------------------------------------------
    max_train_steps: int = 15000
    effective_batch: int = 16             # images per optimizer step
    auto_batch: bool = True               # pick batch size / checkpointing from VRAM
    batch_size: int = 4                   # used when auto_batch = False
    grad_accum: int = 4                   # used when auto_batch = False
    gradient_checkpointing: bool = False  # used when auto_batch = False
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    rank: int = 32                        # LoRA rank
    caption_dropout: float = 0.1
    snr_gamma: float = 5.0                # Min-SNR loss weighting (0 = off)
    mixed_precision: str = "auto"         # auto | no | fp16 | bf16
    num_workers: int = 0 if os.name == "nt" else 2
    seed: int = 42
    log_steps: int = 50
    checkpoint_steps: int = 1000
    keep_checkpoints: int = 2
    validation_steps: int = 1000          # sample images per view; 0 = off
    validation_inference_steps: int = 25
    validation_prompts: list = field(default_factory=lambda: list(DEFAULT_VALIDATION_PROMPTS))

    # ---- derived paths ----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return Path(self.data_root) / "raw"

    @property
    def prepared_dir(self) -> Path:
        return Path(self.data_root) / ("prepared_smoketest" if self.smoke_test else "prepared")


CFG = Config()
# ##########################################################################
#                          end of CONFIG section
# ##########################################################################

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class DatasetError(Exception):
    """A dataset is missing or unreadable; the message says how to fix it."""


# ==========================================================================
# Small helpers
# ==========================================================================
def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "unknown", "-1"} else text


def _skin_type(value) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return ""
    return f"Fitzpatrick skin type {number}" if 1 <= number <= 6 else ""


def _join(*parts: str) -> str:
    return ", ".join(p for p in parts if p)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def _caption(spec: DatasetSpec, name: str, category: str = "", site: str = "",
             skin: str = "") -> str:
    return _join(
        f"{spec.caption_prefix} {name}",
        category if spec.use_category else "",
        site if spec.use_body_site else "",
        skin if spec.use_skin_type else "",
    )


def _read_table(path: Path):
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python")  # sniffs comma or tab
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _find_file(folder: Path, names: list[str], prefix: str | None = None) -> Path | None:
    wanted = {n.lower() for n in names}
    files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
    for p in files:
        if p.name.lower() in wanted:
            return p
    if prefix:
        for p in files:
            if p.name.lower().startswith(prefix) and p.suffix.lower() != ".zip":
                return p
    return None


def _index_images(folder: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }


def extract_archives(folder: Path, skip_words=("segmentation", "test")) -> None:
    """Unzip every archive under `folder` (nested ones too), once."""
    for _ in range(3):
        extracted_any = False
        for archive in sorted(folder.rglob("*.zip")):
            marker = archive.with_name(archive.name + ".extracted")
            if marker.exists() or any(w in archive.name.lower() for w in skip_words):
                continue
            print(f"  extracting {archive.name} ...")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(archive.with_suffix(""))
            marker.write_text("ok")
            extracted_any = True
        if not extracted_any:
            break


# ==========================================================================
# Step 2: tidy loose files into data/raw/<dataset>/
# ==========================================================================
def organize_layout(cfg: Config) -> None:
    for name, patterns in ADOPT_PATTERNS.items():
        spec = DATASETS[name]
        target = cfg.raw_dir / spec.folder
        if target.is_dir() and any(target.iterdir()):
            continue  # already set up
        loose = []
        for pattern in patterns:
            loose.extend(sorted(Path(cfg.data_root).glob(pattern)))
        if not loose:
            continue
        target.mkdir(parents=True, exist_ok=True)
        for item in loose:
            print(f"Moving {item.name} -> {target}")
            shutil.move(str(item), str(target / item.name))
        if name == "pad_ufes_20":  # its metadata.csv sits at the top level
            meta = Path(cfg.data_root) / "metadata.csv"
            if meta.exists():
                shutil.move(str(meta), str(target / "metadata.csv"))


# ==========================================================================
# Step 3: download Fitzpatrick17k images
# ==========================================================================
def _fitz_rows(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    qc = df["qc"].fillna("").astype(str).str.strip()
    return df[~qc.str.startswith(FITZ_BAD_QC)].reset_index(drop=True)


def download_fitzpatrick(cfg: Config) -> None:
    import requests
    from tqdm import tqdm

    spec = DATASETS["fitzpatrick17k"]
    folder = cfg.raw_dir / spec.folder
    image_dir = folder / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "fitzpatrick17k.csv"

    if not csv_path.exists():
        print(f"Downloading {FITZ_CSV_URL}")
        try:
            response = requests.get(FITZ_CSV_URL, timeout=60)
            response.raise_for_status()
            csv_path.write_bytes(response.content)
        except Exception as error:
            raise DatasetError(
                f"Could not download the Fitzpatrick17k CSV ({error}). Download it "
                f"from {FITZ_CSV_URL} and save it as {csv_path}, then re-run."
            )

    df = _fitz_rows(csv_path)
    failures_path = folder / "download_failures.tsv"
    known_failed = set()
    if failures_path.exists() and not cfg.retry_failed_downloads:
        known_failed = {ln.split("\t")[0] for ln in failures_path.read_text().splitlines() if ln}

    have = set(_index_images(folder))
    todo = [
        (row.md5hash, row.url)
        for row in df.itertuples()
        if isinstance(row.url, str) and row.md5hash not in have and row.md5hash not in known_failed
    ]
    if cfg.fitz_limit:
        todo = todo[: cfg.fitz_limit]
    print(
        f"Fitzpatrick17k: {len(df)} usable rows, {len(have)} images on disk, "
        f"{len(todo)} to download"
    )
    if not todo:
        return

    def fetch(item) -> tuple[str, bool]:
        md5, url = item
        parsed = urlparse(url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Medsynth research downloader)",
            "Referer": f"{parsed.scheme}://{parsed.netloc}/",
        }
        for attempt in range(3):
            try:
                reply = requests.get(url, headers=headers, timeout=(10, 60))
                if reply.status_code == 200 and reply.content:
                    Image.open(BytesIO(reply.content)).verify()  # a real image?
                    (image_dir / f"{md5}.jpg").write_bytes(reply.content)
                    return md5, True
                if reply.status_code in (403, 404, 410):
                    break
            except Exception:
                pass
            time.sleep(2**attempt)
        return md5, False

    failed = []
    with ThreadPoolExecutor(max_workers=cfg.fitz_workers) as pool:
        for (md5, ok), (_, url) in zip(
            tqdm(pool.map(fetch, todo), total=len(todo), desc="downloading"), todo
        ):
            if not ok:
                failed.append(f"{md5}\t{url}")
    if failed:
        old = failures_path.read_text().splitlines() if failures_path.exists() else []
        failures_path.write_text("\n".join(sorted(set(old + failed))))
    print(
        f"Downloaded {len(todo) - len(failed)} images, {len(failed)} failed "
        "(dead links are normal; they are not retried on later runs)."
    )


# ==========================================================================
# Step 4: build captions and prepare images
# ==========================================================================
def build_ham10000(spec: DatasetSpec, folder: Path):
    extract_archives(folder)
    meta = _find_file(
        folder, ["HAM10000_metadata.csv", "HAM10000_metadata"], prefix="ham10000_metadata"
    )
    if meta is None:
        raise DatasetError(f"HAM10000_metadata(.csv) not found under {folder}")
    images = _index_images(folder)
    df = _read_table(meta)

    records, missing = [], 0
    for row in df.to_dict("records"):
        code = _clean(row.get("dx")).lower()
        if code not in spec.label_map:
            continue
        src = images.get(_clean(row.get("image_id")))
        if src is None:
            missing += 1
            continue
        name, category = spec.label_map[code]
        caption = _caption(spec, name, category, _clean(row.get("localization")))
        records.append({"key": _safe(row["image_id"]), "src": src,
                        "caption": caption, "label": code})
    return records, missing


def build_pad_ufes_20(spec: DatasetSpec, folder: Path):
    extract_archives(folder)
    meta = _find_file(folder, ["metadata.csv"])
    if meta is None:
        raise DatasetError(f"metadata.csv not found under {folder}")
    images = _index_images(folder)
    df = _read_table(meta)
    skin_col = "fitspatrick" if "fitspatrick" in df.columns else "fitzpatrick"

    records, missing = [], 0
    for row in df.to_dict("records"):
        code = _clean(row.get("diagnostic")).upper()
        if code not in spec.label_map:
            continue
        stem = Path(_clean(row.get("img_id"))).stem
        src = images.get(stem)
        if src is None:
            missing += 1
            continue
        name, category = spec.label_map[code]
        caption = _caption(
            spec, name, category,
            _clean(row.get("region")).lower(), _skin_type(row.get(skin_col)),
        )
        records.append({"key": _safe(stem), "src": src,
                        "caption": caption, "label": code.lower()})
    return records, missing


def build_fitzpatrick17k(spec: DatasetSpec, folder: Path):
    extract_archives(folder)
    csv_path = _find_file(folder, ["fitzpatrick17k.csv"])
    if csv_path is None:
        raise DatasetError(f"fitzpatrick17k.csv not found under {folder}")
    images = _index_images(folder)
    df = _fitz_rows(csv_path)

    records, missing = [], 0
    for row in df.to_dict("records"):
        label = _clean(row.get("label")).lower()
        if not label:
            continue
        src = images.get(_clean(row.get("md5hash")))
        if src is None:
            missing += 1
            continue
        caption = _caption(
            spec, label,
            _clean(row.get("three_partition_label")).lower(),
            skin=_skin_type(row.get("fitzpatrick_scale")),
        )
        records.append({"key": _safe(row["md5hash"]), "src": src,
                        "caption": caption, "label": label})
    return records, missing


BUILDERS = {
    "ham10000": build_ham10000,
    "pad_ufes_20": build_pad_ufes_20,
    "fitzpatrick17k": build_fitzpatrick17k,
}


def _cap_per_class(records: list[dict], cap: int, seed: int) -> list[dict]:
    if cap <= 0:
        return records
    rng = random.Random(seed)
    groups = defaultdict(list)
    for rec in sorted(records, key=lambda r: r["key"]):
        groups[rec["label"]].append(rec)
    kept = []
    for label in sorted(groups):
        items = groups[label]
        if len(items) > cap:
            rng.shuffle(items)
            items = items[:cap]
        kept.extend(items)
    return kept


def _process_image(job) -> str:
    src, out_path, resolution, min_side = job
    if out_path.exists():
        return "ok"
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if min(im.size) < min_side:
                return "small"
            im = ImageOps.fit(im, (resolution, resolution), Image.Resampling.LANCZOS)
            im.save(out_path, quality=95)
        return "ok"
    except Exception as error:  # corrupt / unreadable file
        print(f"  skipped {Path(src).name}: {error}")
        return "bad"


def prepare(cfg: Config) -> Path:
    from tqdm import tqdm

    out = cfg.prepared_dir
    (out / "images").mkdir(parents=True, exist_ok=True)

    all_records, summary, problems = [], {}, []
    for spec in DATASETS.values():
        print(f"[{spec.name}] {spec.view} view")
        folder = cfg.raw_dir / spec.folder
        if not folder.is_dir():
            problems.append((spec, f"folder {folder} not found"))
            print("  folder not found")
            continue
        try:
            records, missing = BUILDERS[spec.name](spec, folder)
        except DatasetError as error:
            problems.append((spec, str(error)))
            print(f"  {error}")
            continue

        records = _cap_per_class(records, spec.max_per_class, cfg.seed)
        if cfg.limit_per_dataset:
            random.Random(cfg.seed).shuffle(records)
            records = records[: cfg.limit_per_dataset]
        if len(records) < cfg.min_images_per_dataset:
            problems.append(
                (spec, f"only {len(records)} usable images found "
                       f"(need at least {cfg.min_images_per_dataset})")
            )
            print(f"  only {len(records)} usable images")
            continue
        for rec in records:
            rec["dataset"] = spec.name
            rec["min_side"] = spec.min_side
        summary[spec.name] = {"images": len(records), "rows_without_image_file": missing}
        print(f"  {len(records)} images kept ({missing} metadata rows had no image file)")
        all_records.extend(records)

    if problems and (cfg.require_all_datasets or not all_records):
        lines = ["", "Cannot start training: some datasets are not ready.", ""]
        for spec, why in problems:
            lines.append(f"  - {spec.name}: {why}")
            lines.append("      " + MANUAL_HELP[spec.name].format(folder=cfg.raw_dir / spec.folder))
        lines += ["", "Fix the above and run `python train.py` again "
                      "(set require_all_datasets = False in CONFIG to train on a subset)."]
        sys.exit("\n".join(lines))
    for spec, why in problems:
        print(f"WARNING: training without {spec.name} ({why})")

    jobs = []
    for rec in all_records:
        rec["file_name"] = f"images/{rec['dataset']}_{rec['key']}.jpg"
        jobs.append((rec["src"], out / rec["file_name"], cfg.resolution, rec["min_side"]))
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        status = list(tqdm(pool.map(_process_image, jobs), total=len(jobs), desc="resizing"))
    dropped = Counter(status)
    if dropped["small"] or dropped["bad"]:
        print(f"  dropped {dropped['small']} too-small and {dropped['bad']} unreadable images")
    all_records = [r for r, s in zip(all_records, status) if s == "ok"]

    rng = random.Random(cfg.seed)
    train_rows, held_rows = [], []
    for rec in all_records:
        row = {"file_name": rec["file_name"], "text": rec["caption"],
               "dataset": rec["dataset"], "view": DATASETS[rec["dataset"]].view,
               "label": rec["label"]}
        (held_rows if rng.random() < cfg.holdout else train_rows).append(row)

    def dump(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump(out / "metadata.jsonl", train_rows)
    dump(out / "heldout.jsonl", held_rows)

    examples = defaultdict(Counter)  # one example prompt per (dataset, class)
    for row in train_rows:
        examples[(row["dataset"], row["label"])][row["text"]] += 1
    with open(out / "prompt_examples.txt", "w", encoding="utf-8") as fh:
        for key in sorted(examples):
            fh.write(examples[key].most_common(1)[0][0] + "\n")

    counts = Counter((r["dataset"], r["label"]) for r in train_rows)
    (out / "prepare_summary.json").write_text(
        json.dumps(
            {
                "datasets": summary,
                "train_images": len(train_rows),
                "heldout_images": len(held_rows),
                "per_class": {f"{d}/{l}": n for (d, l), n in sorted(counts.items())},
                "resolution": cfg.resolution,
            },
            indent=2,
        )
    )
    print(f"\nPrepared {len(train_rows)} training images "
          f"({len(held_rows)} held out) in {out}")
    return out


# ==========================================================================
# Step 5: train
# ==========================================================================
class CaptionDataset(Dataset):
    """Reads prepared/metadata.jsonl. Module-level so DataLoader workers can
    pickle it on Windows. Augmentation follows each dataset's spec."""

    def __init__(self, root: Path, rows: list[dict], tokenizer, resolution: int,
                 caption_dropout: float):
        self.root = Path(root)
        self.rows = rows
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.caption_dropout = caption_dropout

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        spec = DATASETS[row["dataset"]]
        with Image.open(self.root / row["file_name"]) as im:
            im = im.convert("RGB")
        size = (self.resolution, self.resolution)
        if im.size != size:
            im = ImageOps.fit(im, size, Image.Resampling.LANCZOS)

        if spec.hflip and random.random() < 0.5:
            im = ImageOps.mirror(im)
        if spec.vflip and random.random() < 0.5:
            im = ImageOps.flip(im)
        if spec.rot90:
            im = im.rotate(90 * random.randint(0, 3))

        array = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(array).permute(2, 0, 1).contiguous()

        caption = "" if random.random() < self.caption_dropout else row["text"]
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def check_cuda(cfg: Config) -> bool:
    """Make sure training runs on an NVIDIA GPU. Fails fast with a fix instead
    of silently training on the CPU (which would take weeks)."""
    if torch is None:
        sys.exit("PyTorch is not installed. See README.md, section 'Install'.")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(
            f"CUDA GPU: {props.name} | {props.total_memory / 2**30:.1f} GB VRAM | "
            f"torch {torch.__version__} (CUDA {torch.version.cuda})"
        )
        return True

    if torch.version.cuda is None:
        reason = "the installed PyTorch is a CPU-only build"
        fix = (
            "Reinstall the CUDA build of PyTorch. Pick the command for your driver at\n"
            "  https://pytorch.org/get-started/locally  (Stable -> Pip -> CUDA), e.g.\n"
            "  pip uninstall -y torch torchvision\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    else:
        reason = "PyTorch has CUDA support but no usable NVIDIA GPU/driver was found"
        fix = "Check that `nvidia-smi` works, update the NVIDIA driver, and restart."
    message = f"CUDA is not available: {reason}.\n{fix}"
    if cfg.allow_cpu:
        print("WARNING: " + message + "\nContinuing on CPU because allow_cpu = True (very slow).")
        return False
    sys.exit(message + "\n\n(Real training needs an NVIDIA GPU.)")


def choose_batch_settings(cfg: Config) -> Config:
    """Keep the effective batch size fixed but fit it into the GPU's memory."""
    if not cfg.auto_batch or torch is None or not torch.cuda.is_available():
        return cfg
    vram = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
    # Estimates for SD 1.5 LoRA at 512 px with mixed precision.
    if vram >= 20:
        batch, checkpointing = 4, False
    elif vram >= 14:
        batch, checkpointing = 2, False
    elif vram >= 10:
        batch, checkpointing = 2, True
    else:
        batch, checkpointing = 1, True
        if vram < 7:
            print(f"WARNING: only {vram:.1f} GB VRAM; training may run out of memory.")
    accum = max(1, cfg.effective_batch // batch)
    print(f"Auto batch settings for {vram:.0f} GB: batch {batch} x accumulation {accum} "
          f"= {batch * accum} images/step, gradient checkpointing "
          f"{'on' if checkpointing else 'off'}")
    return replace(cfg, batch_size=batch, grad_accum=accum,
                   gradient_checkpointing=checkpointing)


def _resolve_mixed_precision(choice: str) -> str:
    if choice != "auto":
        return choice
    if not torch.cuda.is_available():
        return "no"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def _latest_state(out_dir: Path):
    found = []
    search_dirs = [
        out_dir / "state",
        out_dir / "checkpoints",
        PROJECT_DIR / "model" / "checkpoints",
        PROJECT_DIR / "model" / "state",
        PROJECT_DIR / "model",
    ]
    for s_dir in search_dirs:
        if not s_dir.is_dir():
            continue
        for path in s_dir.iterdir():
            if not path.is_dir():
                continue
            name = path.name
            step_num = None
            if name.startswith("lora-step-"):
                try:
                    step_num = int(name.replace("lora-step-", ""))
                except ValueError:
                    pass
            elif name.startswith("step-"):
                try:
                    step_num = int(name.replace("step-", ""))
                except ValueError:
                    pass
            if step_num is not None:
                is_full_state = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
                found.append((step_num, 1 if is_full_state else 0, path))
    
    if not found:
        return None
    found.sort(key=lambda x: (x[0], x[1]))
    best_step, _, best_path = found[-1]
    return (best_step, best_path)


def _ensure_lora_weights(out_dir: Path, step: int = 0) -> None:
    lora_target = out_dir / "lora"
    lora_target.mkdir(parents=True, exist_ok=True)
    if (lora_target / "pytorch_lora_weights.safetensors").exists():
        return

    search_paths = []
    if step > 0:
        search_paths.append(out_dir / "checkpoints" / f"lora-step-{step}")
        search_paths.append(PROJECT_DIR / "model" / "checkpoints" / f"lora-step-{step}")

    search_paths.extend([
        PROJECT_DIR / "model" / "lora",
        PROJECT_DIR / "model" / "checkpoints",
        out_dir / "checkpoints",
    ])

    for base in [PROJECT_DIR / "model" / "checkpoints", out_dir / "checkpoints"]:
        if base.is_dir():
            for p in sorted(
                base.glob("lora-step-*"),
                key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
                reverse=True,
            ):
                search_paths.append(p)

    for cand in search_paths:
        if cand.is_file() and cand.name == "pytorch_lora_weights.safetensors":
            src = cand
        elif cand.is_dir() and (cand / "pytorch_lora_weights.safetensors").exists():
            src = cand / "pytorch_lora_weights.safetensors"
        else:
            continue

        shutil.copy(src, lora_target / "pytorch_lora_weights.safetensors")
        print(f"Copied trained LoRA weights from {src} to {lora_target}")
        return


def train(cfg: Config) -> None:
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params, compute_snr
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from torch.utils.data import WeightedRandomSampler
    from tqdm import tqdm
    from transformers import CLIPTextModel, CLIPTokenizer

    data_dir = cfg.prepared_dir
    rows = [json.loads(line) for line in open(data_dir / "metadata.jsonl", encoding="utf-8")
            if line.strip()]
    if not rows:
        sys.exit("metadata.jsonl is empty.")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed_precision = _resolve_mixed_precision(cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum, mixed_precision=mixed_precision
    )
    set_seed(cfg.seed)
    if accelerator.device.type != "cuda" and not cfg.allow_cpu:
        sys.exit(f"Accelerate selected '{accelerator.device.type}' instead of CUDA. "
                 "Run `accelerate config` and choose a single GPU, or check your CUDA install.")
    if accelerator.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on Ampere+
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True          # fixed-size inputs -> autotune
        torch.set_float32_matmul_precision("high")

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        accelerator.mixed_precision, torch.float32
    )
    per_view = Counter(r["view"] for r in rows)
    print(f"Device: {accelerator.device} | precision: {accelerator.mixed_precision} | "
          f"images: {len(rows)} {dict(per_view)}")

    # ---- models ---------------------------------------------------------
    base = cfg.base_model
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")

    for model in (unet, vae, text_encoder):
        model.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder.eval()

    unet.add_adapter(
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if weight_dtype != torch.float32:
        cast_training_params(unet, dtype=torch.float32)  # LoRA weights stay fp32
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params) / 1e6:.1f} M")

    optimizer_kwargs = dict(lr=cfg.learning_rate, betas=(0.9, 0.999),
                            weight_decay=1e-2, eps=1e-8)
    try:  # fused CUDA kernel: fewer launches, faster optimizer step
        optimizer = torch.optim.AdamW(
            lora_params, fused=accelerator.device.type == "cuda", **optimizer_kwargs
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(lora_params, **optimizer_kwargs)

    # Every epoch draws from the three views in the proportions set by
    # DatasetSpec.sample_share, so small datasets are not drowned out.
    counts = Counter(r["dataset"] for r in rows)
    weights = [DATASETS[r["dataset"]].sample_share / counts[r["dataset"]] for r in rows]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(rows), replacement=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    dataset = CaptionDataset(data_dir, rows, tokenizer, cfg.resolution, cfg.caption_dropout)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    steps_per_epoch = max(1, math.ceil(len(loader) / cfg.grad_accum))
    num_epochs = math.ceil(cfg.max_train_steps / steps_per_epoch)
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, loader, lr_scheduler
    )

    # ---- helpers --------------------------------------------------------
    def save_lora(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(accelerator.unwrap_model(unet))
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(path), unet_lora_layers=state, safe_serialization=True
        )

    def log_samples(step: int) -> None:
        try:
            unet.eval()
            pipe = StableDiffusionPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                unet=accelerator.unwrap_model(unet),
                scheduler=DPMSolverMultistepScheduler.from_config(noise_scheduler.config),
                safety_checker=None, feature_extractor=None,
                requires_safety_checker=False,
            )
            pipe.set_progress_bar_config(disable=True)
            sample_dir = out_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            use_autocast = weight_dtype != torch.float32
            for i, prompt in enumerate(cfg.validation_prompts):
                generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed + i)
                with torch.autocast(accelerator.device.type, dtype=weight_dtype,
                                    enabled=use_autocast):
                    image = pipe(
                        prompt, num_inference_steps=cfg.validation_inference_steps,
                        guidance_scale=7.0, height=cfg.resolution, width=cfg.resolution,
                        generator=generator,
                    ).images[0]
                image.save(sample_dir / f"step-{step:06d}-{i}.png")
            print(f"  wrote {len(cfg.validation_prompts)} sample image(s) to {sample_dir}")
        except Exception as error:  # never let a preview kill a long run
            print(f"  (sample generation skipped: {error})")
        finally:
            unet.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_checkpoint(step: int) -> None:
        state_root = out_dir / "state"
        accelerator.save_state(str(state_root / f"step-{step}"))
        save_lora(out_dir / "checkpoints" / f"lora-step-{step}")
        old = sorted(state_root.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
        for stale in old[: max(0, len(old) - cfg.keep_checkpoints)]:
            shutil.rmtree(stale, ignore_errors=True)
        peak = (f", peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                if torch.cuda.is_available() else "")
        print(f"  checkpoint saved at step {step}{peak}")

    # ---- automatic resume ----------------------------------------------
    global_step, first_epoch, resume_batches = 0, 0, 0
    latest = _latest_state(out_dir)
    did_train = False
    if latest is not None:
        global_step, path = latest
        if global_step < cfg.max_train_steps:
            if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
                accelerator.load_state(str(path))
            first_epoch = global_step // steps_per_epoch
            resume_batches = (global_step - first_epoch * steps_per_epoch) * cfg.grad_accum
            print(f"Resuming from {path.name} (step {global_step} of {cfg.max_train_steps})")
        else:
            print(f"Found completed model/checkpoint at step {global_step} in {path}")

    (out_dir / "train_config.json").write_text(
        json.dumps({"config": asdict(cfg), "datasets": {k: asdict(v) for k, v in DATASETS.items()},
                    "num_images": len(rows), "precision": mixed_precision},
                   indent=2, default=str)
    )
    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss,lr\n")

    # ---- training loop --------------------------------------------------
    prediction_type = noise_scheduler.config.prediction_type
    if global_step < cfg.max_train_steps:
        did_train = True
        progress = tqdm(total=cfg.max_train_steps, initial=global_step, desc="training")
        unet.train()
        running_loss, running_n = 0.0, 0

        for epoch in range(first_epoch, num_epochs):
            batches = loader
            if epoch == first_epoch and resume_batches > 0:
                batches = accelerator.skip_first_batches(loader, resume_batches)

            for batch in batches:
                with accelerator.accumulate(unet):
                    pixels = batch["pixel_values"].to(dtype=weight_dtype)
                    with torch.no_grad():
                        latents = vae.encode(pixels).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                        text_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],), device=latents.device,
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    if prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    prediction = unet(noisy_latents, timesteps, text_states,
                                      return_dict=False)[0]

                    if cfg.snr_gamma and cfg.snr_gamma > 0:
                        snr = compute_snr(noise_scheduler, timesteps)
                        weights_t = torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        if prediction_type == "v_prediction":
                            weights_t = weights_t / (snr + 1)
                        else:
                            weights_t = weights_t / snr
                        loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, loss.ndim))) * weights_t
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                running_loss += loss.detach().float().item()
                running_n += 1

                if accelerator.sync_gradients:
                    global_step += 1
                    progress.update(1)
                    progress.set_postfix(loss=f"{running_loss / max(running_n, 1):.4f}")

                    if global_step % cfg.log_steps == 0:
                        with open(log_path, "a") as fh:
                            fh.write(f"{global_step},{running_loss / max(running_n, 1):.5f},"
                                     f"{lr_scheduler.get_last_lr()[0]:.3e}\n")
                        running_loss, running_n = 0.0, 0
                    if global_step % cfg.checkpoint_steps == 0:
                        save_checkpoint(global_step)
                    if cfg.validation_steps and global_step % cfg.validation_steps == 0:
                        log_samples(global_step)
                    if global_step >= cfg.max_train_steps:
                        break
            if global_step >= cfg.max_train_steps:
                break
        progress.close()
        if global_step % cfg.checkpoint_steps != 0:
            save_checkpoint(global_step)  # so a crash during export can resume
    else:
        print(f"Training already reached {global_step} steps, skipping to export.")

    if did_train:
        save_lora(out_dir / "lora")
    else:
        _ensure_lora_weights(out_dir, global_step)

    log_samples(global_step)
    print(f"\nTraining finished at step {global_step}. LoRA saved to {out_dir / 'lora'}")
    accelerator.end_training()

    del unet, vae, text_encoder, optimizer, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================================================
# Step 6: export a standalone pipeline for app.py
# ==========================================================================
def export_pipeline(cfg: Config) -> Path:
    from diffusers import StableDiffusionPipeline

    out_dir = Path(cfg.output_dir)
    lora_dir = out_dir / "lora"
    if not (lora_dir / "pytorch_lora_weights.safetensors").exists():
        _ensure_lora_weights(out_dir)

    if not (lora_dir / "pytorch_lora_weights.safetensors").exists():
        sys.exit(f"No pytorch_lora_weights.safetensors in {lora_dir}. Training did not finish.")

    print(f"Merging {lora_dir} into {cfg.base_model} ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float32,  # merge in fp32 on the CPU, then shrink to fp16
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights(str(lora_dir))
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.to(torch.float16)

    target = out_dir / "pipeline"
    pipe.save_pretrained(str(target), safe_serialization=True)

    examples = cfg.prepared_dir / "prompt_examples.txt"
    if examples.exists():
        shutil.copy(examples, out_dir / "prompt_examples.txt")
    return target


def _marker(cfg: Config) -> Path:
    return Path(cfg.output_dir) / "training_complete.json"


def _model_ready(cfg: Config) -> bool:
    return _marker(cfg).exists() and (Path(cfg.output_dir) / "pipeline" / "model_index.json").exists()


def smoke_config(cfg: Config) -> Config:
    """Tiny end-to-end run (a few minutes) that writes to its own folders."""
    return replace(
        cfg, output_dir=PROJECT_DIR / "medsynth-smoketest",
        limit_per_dataset=40, min_images_per_dataset=10, fitz_limit=60,
        max_train_steps=40, lr_warmup_steps=5, auto_batch=False,
        batch_size=1, grad_accum=1, gradient_checkpointing=True,
        checkpoint_steps=20, keep_checkpoints=1, log_steps=5,
        validation_steps=20, validation_inference_steps=10,
    )


# ==========================================================================
# main: one command does everything
# ==========================================================================
def main() -> None:
    cfg = smoke_config(CFG) if CFG.smoke_test else CFG
    out_dir = Path(cfg.output_dir)

    print("=" * 64)
    print("Medsynth trainer" + ("  [SMOKE TEST]" if cfg.smoke_test else ""))
    for spec in DATASETS.values():
        print(f"  {spec.view:20s} {spec.name:15s} share {spec.sample_share:.0%}")
    print(f"  model folder: {out_dir}")
    print("=" * 64)

    if _model_ready(cfg) and not cfg.retrain:
        info = json.loads(_marker(cfg).read_text())
        print(f"\nA trained model already exists ({info.get('steps')} steps, "
              f"finished {info.get('finished_at')}).\n"
              f"  location: {out_dir / 'pipeline'}\n"
              "Nothing to do. Run `python app.py` to use it.\n"
              "To train again from scratch, set retrain = True in the CONFIG section.")
        return

    check_cuda(cfg)  # fail fast, before any long preparation

    if cfg.retrain:
        print("retrain = True: discarding old checkpoints and starting from scratch.")
        shutil.rmtree(out_dir / "state", ignore_errors=True)
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
        _marker(cfg).unlink(missing_ok=True)
        (out_dir / "train_log.csv").unlink(missing_ok=True)

    organize_layout(cfg)
    if cfg.auto_download_fitzpatrick:
        try:
            download_fitzpatrick(cfg)
        except DatasetError as error:
            print(f"WARNING: {error}")
    prepare(cfg)

    cfg = choose_batch_settings(cfg)
    train(cfg)
    target = export_pipeline(cfg)

    steps = cfg.max_train_steps
    (out_dir / "model_info.json").write_text(json.dumps({
        "base_model": cfg.base_model, "pipeline_dir": str(target.resolve()),
        "lora_dir": str((out_dir / "lora").resolve()), "steps": steps,
        "rank": cfg.rank, "resolution": cfg.resolution,
        "datasets": list(DATASETS), "views": [s.view for s in DATASETS.values()],
    }, indent=2))
    _marker(cfg).write_text(json.dumps({
        "steps": steps, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}))

    print(f"\nDone. Standalone model saved to: {target.resolve()}")
    print("It is permanent: run `python app.py` to use it. No retraining is needed.")


if __name__ == "__main__":
    main()