from pathlib import Path
from diffusers import AutoPipelineForText2Image
from .train_text_to_image_lora import parse_args, run_training
import torch
import json
import shutil

device = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

def create_unified_file(data):
    src1 = Path("data/HAM10000_images_part_1")
    src2 = Path("data/HAM10000_images_part_2")
    dst  = Path("data/ham_lora")

    dst.mkdir(parents=True, exist_ok=True)

    for p in list(src1.glob("*.jpg")) + list(src2.glob("*.jpg")):
        target = dst / p.name
        if not target.exists():
            shutil.copy2(p, target)


    meta_path = dst / "metadata.jsonl"

    rows = 0
    with open(meta_path, "w") as f:
        for _, row in data.iterrows():
            image_id = row["image_id"]
            if not image_id.endswith(".jpg"):
                image_id += ".jpg"

            img_path = dst / image_id
            if not img_path.exists():
                continue

            prompt = row["Prompt"]

            f.write(json.dumps({"file_name": image_id, "text": prompt}) + "\n")
            rows += 1

    print("Wrote", rows, "rows to metadata.jsonl")
    

def train(resolution, lr, epochs, output):
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    TRAIN_DIR = "data/ham_lora"
    OUTPUT_DIR = f"./LoRA Weights/{output}"

    args_list = [
        "--pretrained_model_name_or_path", MODEL_NAME,
        "--train_data_dir", TRAIN_DIR,
        "--caption_column", "text",
        "--resolution", str(resolution),
        "--train_batch_size", "1",
        "--num_train_epochs", str(epochs),
        "--learning_rate", str(lr),
        "--mixed_precision", "fp16",
        "--output_dir", OUTPUT_DIR,
        "--checkpointing_steps", "500",
        "--seed", "42",
    ]

    args = parse_args(args_list)
    run_training(args)
    
    