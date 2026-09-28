from pytorch_fid import fid_score
import torch

fid_value = fid_score.calculate_fid_given_paths(
    ['./data/ham_lora', 'path/to/generated_images'],
    batch_size=50,
    device='cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu',  
    dims=2048
)

print(f"FID: {fid_value:.2f}")
