from functools import lru_cache
from threading import Lock

import gradio as gr


MODEL_ID = "runwayml/stable-diffusion-v1-5"
_generation_lock = Lock()
_active_lora = None


@lru_cache(maxsize=1)
def load_pipeline():
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    return pipeline, device


def set_lora_weights(pipeline, lora_weights):
    global _active_lora

    if lora_weights == _active_lora:
        return
    if _active_lora:
        pipeline.unload_lora_weights()
        _active_lora = None
    if lora_weights:
        pipeline.load_lora_weights(lora_weights)
    _active_lora = lora_weights


def generate_image(prompt, history, lora_weights, steps, guidance_scale, seed):
    history = list(history or [])
    prompt = (prompt or "").strip()
    if not prompt:
        return history, None, ""

    history.append({"role": "user", "content": prompt})
    try:
        import torch

        with _generation_lock:
            pipeline, device = load_pipeline()
            set_lora_weights(pipeline, (lora_weights or "").strip())
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=device).manual_seed(int(seed))

            image = pipeline(
                prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0]
        history.append({"role": "assistant", "content": "Image generated."})
        return history, image, ""
    except Exception as error:
        history.append({"role": "assistant", "content": f"Generation failed: {error}"})
        return history, None, ""


with gr.Blocks(
    title="Medsynth",
    theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    css="""
    .gradio-container { max-width: 1240px !important; }
    .app-header { padding: 1.25rem 0 .5rem; }
    .app-header h1 { margin-bottom: .25rem; }
    .result-panel { border-left: 1px solid var(--border-color-primary); padding-left: 1rem; }
    """,
) as app:
    gr.Markdown(
        "# Medsynth\nDescribe a dermoscopic image to generate. Generations are synthetic and not for diagnosis.",
        elem_classes="app-header",
    )
    with gr.Row():
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(
                type="messages",
                height=490,
                placeholder="Your image-generation conversation will appear here.",
                label="Conversation",
            )
            with gr.Row():
                prompt = gr.Textbox(
                    placeholder="Describe a skin lesion and its clinical context...",
                    label="Prompt",
                    lines=2,
                    scale=8,
                )
                send = gr.Button("Generate", variant="primary", scale=1)
        with gr.Column(scale=4, elem_classes="result-panel"):
            image_output = gr.Image(label="Generated image", type="pil", height=420)
            with gr.Accordion("Generation settings", open=False):
                lora_weights = gr.Textbox(
                    label="LoRA weights path (optional)",
                    placeholder="Path to a trained LoRA adapter",
                )
                steps = gr.Slider(1, 50, value=25, step=1, label="Inference steps")
                guidance_scale = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance scale")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")

    inputs = [prompt, chatbot, lora_weights, steps, guidance_scale, seed]
    outputs = [chatbot, image_output, prompt]
    prompt.submit(generate_image, inputs=inputs, outputs=outputs, show_api=False)
    send.click(generate_image, inputs=inputs, outputs=outputs, show_api=False)


if __name__ == "__main__":
    app.queue().launch()