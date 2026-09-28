import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from pathlib import Path

device = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

def compare_embeddings(path_to_image, prompt):
    def get_clip_embedding(img, processor, model):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu()

    def compute_ssim(img1, img2, size=(256, 256)):
        img1_arr = np.array(img1.resize(size))
        img2_arr = np.array(img2.resize(size))
        s, _ = ssim(img1_arr, img2_arr, channel_axis=2, full=True)
        return float(s)

    generated_image_path = Path(path_to_image)
    TOP_K = 5

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    gen_img = Image.open(generated_image_path).convert("RGB")
    gen_emb = get_clip_embedding(gen_img, processor, model)

    embeddings = []
    paths = []
    dst = Path("./data/ham_lora")
    
    for p in tqdm(list(dst.rglob("*.jpg")), desc="Embedding dataset"):
        try:
            img = Image.open(p).convert("RGB")
        except:
            continue
        emb = get_clip_embedding(img, processor, model)
        embeddings.append(emb)
        paths.append(p)

    embeddings = torch.cat(embeddings, dim=0)
    similarities = (embeddings @ gen_emb.T).squeeze(1)

    topk = torch.topk(similarities, k=TOP_K)
    top_scores = topk.values.tolist()
    top_indices = topk.indices.tolist()

    top_k_paths = []
    top_k_clip = []
    top_k_ssim = []

    for score, idx in zip(top_scores, top_indices):
        candidate_path = paths[idx]
        cand_img = Image.open(candidate_path).convert("RGB")
        ssim_score = compute_ssim(gen_img, cand_img)
        top_k_paths.append(candidate_path)
        top_k_clip.append(score)
        top_k_ssim.append(ssim_score)
        
    def get_prompt_for_path(path, metadata):
        fname = os.path.basename(str(path))
        return metadata.get(fname, "(no prompt found)")

    metadata = {}
    
    meta_path = dst / "metadata.jsonl"
    with open(meta_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            metadata[entry["file_name"]] = entry["text"]

    data_rows = [{
        "path": "GENERATED",
        "clip_score": '-',
        "ssim_score": '-',
        "prompt": prompt
    }]

    for path, clip_val, ssim_val in zip(top_k_paths, top_k_clip, top_k_ssim):
        data_rows.append({
            "path": str(path),
            "clip_score": clip_val,
            "ssim_score": ssim_val,
            "prompt": metadata.get(Path(path).name, "NOT FOUND")
        })

    df = pd.DataFrame(data_rows)
    print(df)

    similar_imgs = [Image.open(p).convert("RGB") for p in top_k_paths]
    n_cols = 1 + len(similar_imgs)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 6))

    axes[0].imshow(gen_img)
    axes[0].set_title("Generated")
    axes[0].axis("off")

    for i, (img, clip_score, ssim_score) in enumerate(
        zip(similar_imgs, top_k_clip, top_k_ssim), start=1
    ):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"Similar #{i}\nCLIP={clip_score:.3f}\nSSIM={ssim_score:.3f}")
        ax.axis("off")

    plt.subplots_adjust(bottom=0.1, wspace=0.3)
    plt.show()