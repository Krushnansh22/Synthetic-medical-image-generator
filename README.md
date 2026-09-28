# Synthetic Medical Image Generation

This project explores two different approaches for generating realistic medical images of skin diseases from structured clinical metadata. We compare:

1. **Method A — Tabular → Text Prompt → Multi-Modal Generative model with LoRA fine-tuning**  
2. **Method B — Direct Conditional Diffusion using tabular feature vectors**

The goal is to evaluate (i) how well each method captures metadata-image relationships, (ii) the realism of the generated images, and (iii) the latency–quality trade-offs under system load.


## Project Overview

Skin lesion datasets such as **ISIC Archive / HAM10000** contain dermoscopic images with clinical metadata:
- Age  
- Sex  
- Anatomical site  
- Diagnosis labels (melanoma, nevus, BCC, etc.)

We build two generative pipelines that use this metadata differently to generate medical images.

## Method A: Tabular → Text Prompt → Stable Diffusion

This is the *natural language–based* approach:

1. Convert each metadata row into a structured text prompt  

2. Fine-tune **Stable Diffusion** on dermoscopic images using **LoRA**.

3. Generate new images from these prompts using the SD text encoder.

## Method B: Direct Conditional Diffusion (Tabular → Embedding → Diffusion Model)

This is the *fully numeric conditional* approach:

1. Encode metadata (diagnosis, site, sex, age) as a numeric feature vector.  
2. Pass this vector through a small MLP to obtain a conditioning embedding.  
3. Inject this embedding into a **diffusion model**  
4. Train a **custom conditional diffusion model**.

## Evaluation Metrics

We compare the two methods using:

### **Realism**
- FID  
- KID  
- MS-SSIM diversity score  

### **Metadata Faithfulness**
- CLIPScore (prompt ↔ image similarity)  
- Attribute agreement tests (does changing metadata change the image?)  

### **System Performance**
- Image generation latency  
- Quality–latency trade-offs  
- Parameter tuning under load (e.g., number of diffusion steps)

## Repository Structure

## Dataset

We use the **HAM10000 / ISIC** public dataset:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000  
Images: ~10,000 dermoscopic RGB images  
Metadata: age, sex, anatomical site, diagnosis  
License: CC BY-NC (research use)
