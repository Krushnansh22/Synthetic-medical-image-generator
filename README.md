# Medsynth

Medsynth is a research project for generating synthetic dermoscopic skin-lesion images from text prompts. Its Gradio chat interface runs Stable Diffusion 1.5 and can optionally load a locally trained LoRA adapter. Generated images are synthetic and must not be used for diagnosis or treatment decisions.

## Setup

Use Python **3.11.0**. From the project directory, create and activate a virtual environment, then install the pinned dependencies:

```powershell
uv venv env
.\env\Scripts\activate
uv pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

## Run The App

```powershell
python main.py
```

Open the local URL printed by Gradio. The Stable Diffusion 1.5 model downloads on the first generation, so an internet connection is required initially. A CUDA-enabled NVIDIA GPU is recommended; CPU inference is supported but slow. Enter a prompt and select **Generate**. The settings panel controls inference steps, guidance scale, seed, and an optional local LoRA weights path.

## Project Structure

```text
Medsynth/
|-- main.py                              # Gradio app and image-generation callback
|-- requirements.txt                     # Pinned Python dependencies
|-- data/                                # HAM10000 metadata and images for training
|-- utils/
|   |-- preprocessing.py                 # Metadata preparation
|   `-- promptbased/
|       |-- PromptBuilder.py              # Metadata-to-text prompt templates
|       |-- train.py                      # Dataset preparation and training helper
|       `-- train_text_to_image_lora.py   # Diffusers LoRA training implementation
|-- LoRA Weights/                         # Local fine-tuned adapters (created by training)
`-- results/                              # Generated outputs (created as needed)
```

The Gradio app is the default workflow. `main.py` loads one base diffusion pipeline on the first image request and reuses it for subsequent generations. CUDA uses half precision; CPU uses full precision. When the optional LoRA path changes, its weights are swapped on the existing pipeline rather than loading another copy of the base model. Generations are serialized so adapter changes cannot interfere with an image in progress.

The separate fine-tuning utilities require the HAM10000 metadata and image folders under `data/`, plus a CUDA-capable GPU with substantial memory. They are not needed to run the web app. The Conda environment files and standalone analysis/queue experiments were removed; use `requirements.txt` for setup.
