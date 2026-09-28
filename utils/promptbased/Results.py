import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoPipelineForText2Image
from .PromptBuilder import prompts
import torch
import time
import random
import numpy as np

device = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
    
def random_prompt():
    dx_list = ["melanoma", "nevus", "basal cell carcinoma", "seborrheic keratosis"]
    site_list = ["back", "arm", "leg", "chest", "neck", "face"]
    sex_list = ["male", "female"]
    age_range = range(20, 85)
    categories = ["full_metadata", "no_sex", "no_localization", "dx_only"]
    selected_category = np.random.choice(categories, p=[0.7, 0.1, 0.1, 0.1])
    prompt_template = random.choice(prompts[selected_category])

    dx = random.choice(dx_list)
    site = random.choice(site_list)
    sex = random.choice(sex_list)
    age = random.choice(list(age_range))

    filled_prompt = (
        prompt_template
        .replace("{dx}", dx)
        .replace("{site}", site)
        .replace("{sex}", sex)
        .replace("{age}", str(age))
    )

    return filled_prompt    

def sample_base(prompt, num_inference_steps, guidance_scale):
    pipeline_base = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
    ).to(device)

    pipeline_base.safety_checker=None
    
    out = pipeline_base(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    image_base = out.images[0]
    plt.imsave('./results/prompt based/base_model.png', image_base)
    return image_base

def sample_finetuned(prompt, LoRA_weights, num_inference_steps, guidance_scale,generate=None):
    pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)

    pipeline.load_lora_weights(
        LoRA_weights,
        weight_name="pytorch_lora_weights.safetensors"
    )
    
    pipeline.safety_checker = None
    
    if generate:
        times = []
        for i in range(generate):
            prompt = random_prompt()
            start = time.time()
            image = pipeline(prompt, 
                             num_inference_steps=num_inference_steps, 
                             guidance_scale=guidance_scale
                             ).images[0]
            plt.imsave(f'./results/prompt based/512/image #{i}.png', image)
            end = time.time()
            times.append(end - start)
        
        print(f'on average took {sum(times[1:])/len(times[1:])} to generate an image')
    
        return times
        
    image = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    plt.imsave('./results/prompt based/finetuned_model.png', image)
    return image

def display_results(base_image, finetuned_image):
    print('\nimages saved to results/prompt based')
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].imshow(finetuned_image)
    axes[0].set_title('After Fine-Tuning')
    axes[0].axis('off')
    axes[1].imshow(base_image)
    axes[1].set_title('Before Fine-Tuning')
    axes[1].axis('off')
    plt.suptitle('Sampling Comparison', fontweight = 'bold', fontsize=14)
    plt.tight_layout()
    plt.show()