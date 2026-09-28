import torch
import torch.nn as nn
import json
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm

class LoRALinear(nn.Module):
    def __init__(self, base:nn.Linear, r:int, alpha:int = None):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        if self.alpha:
            self.scaling = alpha / r
        else:
            self.scaling = r / r
        
        self.base.requires_grad_(False)
        
        self.in_features = self.base.in_features
        self.out_features = self.base.out_features
        
        self.A = nn.Parameter(torch.zeros((self.out_features, self.r)), requires_grad=True)
        self.B = nn.Parameter(torch.randn((self.r, self.in_features)), requires_grad=True)
        
    def forward(self, x):
        # x.shape = (batch, in_features)
        frozen_output = self.base(x)
        B_output = x @ self.B.T # (batch, self.r)
        LoRA_output = B_output @ self.A.T # (batch, self.out)
        output = frozen_output + self.scaling * LoRA_output
        return output


class LoRADataset(Dataset):
    def __init__(self, data_path, transformation = None):
        super().__init__()
        self.data_path = data_path
        self.data = []
        self.transformation = transformation
        self.metadata = os.path.join(data_path, 'metadata.jsonl')
        
        with open(self.metadata, 'r') as f:
            num = 0
            for line in f:
                if num < 5000:
                    image_caption_pair = json.loads(line)
                    self.data.append(image_caption_pair)
                    num += 1
                
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]['file_name']
        caption = self.data[index]['text']
        image_path = os.path.join(self.data_path, image)
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transformation:
            image = self.transformation(image)
        
        return {'pixel_values':image, 'text': caption}
        


def training_loop(r, learning_rate, data_path, batch_size, resolution, num_workers, epochs, alpha = None):
    model_id = "runwayml/stable-diffusion-v1-5"

    vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet')
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder')
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')

    device = ('cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu')

    vae.to(device)
    text_encoder.to(device)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    vae.eval()
    text_encoder.eval()
    unet.train()
    
    '''
    
    for name, module in unet.named_modules():
        if hasattr(module, 'to_q'):
            module.to_q = LoRALinear(module.to_q, r, alpha)
        if hasattr(module, 'to_k'):
            module.to_k = LoRALinear(module.to_k, r, alpha)
        if hasattr(module, 'to_v'):
            module.to_v = LoRALinear(module.to_v, r, alpha)
        if hasattr(module, "to_out"):
            out_proj = module.to_out
            if isinstance(out_proj, nn.Linear):
                module.to_out = LoRALinear(out_proj, r, alpha)
            elif isinstance(out_proj, nn.Sequential) and len(out_proj) > 0 and isinstance(out_proj[0], nn.Linear):
                out_proj[0] = LoRALinear(out_proj[0], r, alpha)
            elif isinstance(out_proj, nn.ModuleList) and len(out_proj) > 0 and isinstance(out_proj[0], nn.Linear):
                out_proj[0] = LoRALinear(out_proj[0], r, alpha)
    
    '''
    
    if alpha is None:
        alpha = r  # same as their lora_alpha=args.rank

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    
    unet.add_adapter(lora_config)
    unet.to(device)
    unet.train()
                
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, weight_decay=1e-2)
    lossFN = nn.MSELoss()
    
    transformation  = transforms.Compose([
                    transforms.Resize((resolution,resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    
    dataset = LoRADataset(data_path, transformation)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch['pixel_values'] = batch['pixel_values'].to(device, dtype=torch.float32)
            
            with torch.no_grad():
                latents = vae.encode(batch['pixel_values']).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            batch_size_current = batch['pixel_values'].shape[0]
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (batch_size_current,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            with torch.no_grad():
                tokenized = tokenizer(
                    batch['text'], return_tensors = 'pt', 
                    padding = True, 
                    truncation=True)
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                encoder_hidden_states = text_encoder(input_ids=tokenized['input_ids'], 
                                                     attention_mask=tokenized.get('attention_mask', None)).last_hidden_state
                
            epsilon_hat = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            loss = lossFN(epsilon_hat, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    '''
    lora_state_dict = {}
    
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[name] = module.state_dict()
    
    torch.save(lora_state_dict, "lora_weights.pt")
    
    peft_state_dict = get_peft_model_state_dict(unet)
    torch.save(peft_state_dict, "lora_weights.pt")
    print("Saved LoRA weights to lora_weights.pt")
    '''
    
    unet.save_attn_procs("lora")  # folder name, not .pt
    print("Saved LoRA weights to lora")
            

def load_lora_weights(unet, lora_path):
    lora_state_dict = torch.load(lora_path, map_location='cpu')
    
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state_dict:
            module.load_state_dict(lora_state_dict[name])
            

def make_pipeline_with_lora(lora_path="lora"):
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.safety_checker = None

    # Load LoRA weights saved with `save_attn_procs`
    pipe.unet.load_attn_procs(lora_path)

    pipe.to(device)
    return pipe



if __name__ == '__main__':
    training_loop(
        r=4,
        learning_rate=1e-4,
        data_path='data/ham_lora',
        batch_size=1,
        resolution=512,
        num_workers=6,
        epochs=1,
        alpha=8
    )

    pipe = make_pipeline_with_lora("lora")
    image = pipe(
        "Dermoscopy image of a melanoma.",
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    image.save("out.png")
