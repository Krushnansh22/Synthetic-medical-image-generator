import os
from datetime import datetime
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread

# This app only ever uses the model trained by train.py. Never touch the Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
from PIL import Image
from utils import chat_history


# The fine-tuned model lives in the project folder (written by train.py) and is
# loaded from disk once per app start. Override with MEDSYNTH_MODEL_DIR.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get("MEDSYNTH_MODEL_DIR", PROJECT_DIR / "medsynth-model" / "pipeline")
)
_generation_lock = Lock()

# Prompt prefixes that match the captions used during training.
IMAGE_TYPES = {
    "Dermoscopy": "dermoscopy image of",
    "Smartphone": "smartphone photo of",
    "Clinical photo": "clinical photo of",
}

# Note: no anatomy terms (face, nose, lips, hands...) here on purpose. The
# clinical datasets contain lesions on those body sites.
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, 3d render, CGI, "
    "blurry, low resolution, oversaturated, deformed, "
    "watermark, text, logo, food, landscape"
)

TITLE_MAX_CHARS = 32

PENDING_HTML = """
<div class="ms-pending" role="status" aria-live="polite">
  <div class="ms-pending__frame"><div class="ms-pending__scan"></div></div>
  <span class="ms-pending__label">Generating image</span>
</div>
"""


def model_is_available():
    return (MODEL_DIR / "model_index.json").is_file()


@lru_cache(maxsize=1)
def load_pipeline():
    """Load the fine-tuned pipeline from disk (cached for the app's lifetime)."""
    if not model_is_available():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_DIR}. Run train.py first "
            "(see README.md), or set MEDSYNTH_MODEL_DIR to the folder that "
            "contains model_index.json."
        )

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )
    pipeline = pipeline.to(device)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Loaded {MODEL_DIR} on CUDA GPU: {torch.cuda.get_device_name(0)} (FP16)")
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)
    else:
        print(f"Loaded {MODEL_DIR} on CPU (slow). Install a CUDA build of PyTorch for speed.")

    return pipeline, device


def build_medical_prompt(prompt, medical_prompt_enabled, image_type="Dermoscopy"):
    if not medical_prompt_enabled:
        return prompt

    prefix = IMAGE_TYPES.get(image_type, IMAGE_TYPES["Dermoscopy"])
    if any(prompt.lower().startswith(p) for p in IMAGE_TYPES.values()):
        return prompt  # already written in the training-caption style
    return f"{prefix} {prompt}"


def _short_title(title):
    title = " ".join((title or "Untitled").split())
    if len(title) > TITLE_MAX_CHARS:
        title = title[: TITLE_MAX_CHARS - 1].rstrip() + "…"
    return title


def _short_stamp(value):
    """'2026-09-29 14:27:03' -> '29 Sep, 14:27' (falls back to the raw text)."""
    try:
        return datetime.fromisoformat(str(value)).strftime("%d %b, %H:%M")
    except ValueError:
        return str(value)[:16]


def _conversation_choices():
    # Two-line labels: the title on the first line, the time on the second.
    # The CSS uses `white-space: pre-line` so the newline is honoured.
    return [
        (
            f"{_short_title(conversation['title'])}\n"
            f"{_short_stamp(conversation['updated_at'])}",
            conversation["id"],
        )
        for conversation in chat_history.list_conversations()
    ]


def _render_conversation(conversation_id, pending=False):
    rendered_messages = []
    for message in chat_history.get_messages(conversation_id):
        rendered_messages.append(
            {"role": message["role"], "content": message["content"]}
        )

        if message["image"] is not None:
            image = Image.open(BytesIO(message["image"])).copy()
            rendered_messages.append(
                {
                    "role": "assistant",
                    "content": gr.Image(
                        value=image,
                        show_label=False,
                        interactive=False,
                        container=False,
                    ),
                }
            )

    if pending:
        # Transient placeholder, never written to the chat history.
        rendered_messages.append(
            {"role": "assistant", "content": gr.HTML(value=PENDING_HTML)}
        )

    return rendered_messages


def _history_update(selected_id=None):
    choices = _conversation_choices()
    conversation_ids = {value for _, value in choices}
    if selected_id not in conversation_ids:
        selected_id = choices[0][1] if choices else None
    return gr.update(choices=choices, value=selected_id), selected_id


def start_conversation():
    conversation_id = chat_history.create_conversation()
    history_update, _ = _history_update(conversation_id)
    return [], "", conversation_id, history_update


def select_conversation(conversation_id):
    return _render_conversation(conversation_id), conversation_id


def remove_conversation(conversation_id):
    chat_history.delete_conversation(conversation_id)
    choices = _conversation_choices()
    if not choices:
        chat_history.create_conversation()
        choices = _conversation_choices()

    selected_id = choices[0][1]
    return (
        _render_conversation(selected_id),
        selected_id,
        gr.update(choices=choices, value=selected_id),
    )


def generate_image(
    prompt,
    conversation_id,
    image_type,
    steps,
    guidance_scale,
    seed,
    medical_prompt_enabled,
    negative_prompt,
):
    prompt = (prompt or "").strip()
    if not prompt:
        yield _render_conversation(conversation_id), "", conversation_id, gr.update()
        return

    if not conversation_id:
        conversation_id = chat_history.create_conversation()

    chat_history.add_message(conversation_id, "user", prompt)
    history_update, _ = _history_update(conversation_id)
    yield (
        _render_conversation(conversation_id, pending=True),
        "",
        conversation_id,
        history_update,
    )

    try:
        import torch

        with _generation_lock:
            pipeline, device = load_pipeline()

            generator = None
            if seed is not None and int(seed) >= 0:
                generator = torch.Generator(device=device).manual_seed(int(seed))

            enhanced_prompt = build_medical_prompt(
                prompt,
                medical_prompt_enabled,
                image_type,
            )
            negative_prompt = (negative_prompt or "").strip() or None
            image = pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0]

        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        chat_history.add_message(
            conversation_id,
            "assistant",
            "Image generated.",
            image=image_buffer.getvalue(),
            image_mime="image/png",
        )
    except Exception as error:
        chat_history.add_message(
            conversation_id,
            "assistant",
            f"Generation failed: {error}",
        )

    history_update, _ = _history_update(conversation_id)
    yield (
        _render_conversation(conversation_id),
        "",
        conversation_id,
        history_update,
    )


initial_choices = _conversation_choices()
if not initial_choices:
    chat_history.create_conversation()
    initial_choices = _conversation_choices()
initial_conversation_id = initial_choices[0][1]


# --------------------------------------------------------------------------
# Look & feel
# --------------------------------------------------------------------------

HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap">
"""

APP_CSS = """
/* ===== 1. Design tokens (one source of truth, light + dark) ============== */
.gradio-container.gradio-container {
    --font: "Instrument Sans", ui-sans-serif, system-ui, sans-serif;
    --ms-display: "Bricolage Grotesque", "Instrument Sans", ui-sans-serif, system-ui, sans-serif;

    --ms-bg: #edf2f0;
    --ms-side: #f7faf9;
    --ms-surface: #ffffff;
    --ms-surface-2: #f0f5f3;
    --ms-line: #d9e2df;
    --ms-line-strong: #bccac5;
    --ms-text: #10201c;
    --ms-muted: #566761;
    --ms-accent: #0f766e;
    --ms-accent-strong: #115e59;
    --ms-accent-rgb: 15 118 110;
    --ms-on-accent: #ffffff;
    --ms-danger: #b91c1c;
    --ms-danger-rgb: 185 28 28;
    --ms-shadow: 0 1px 2px rgb(16 32 28 / .05), 0 10px 28px -14px rgb(16 32 28 / .22);

    --ms-r-sm: 10px;
    --ms-r-md: 14px;
    --ms-r-lg: 20px;
    --ms-sidebar: clamp(248px, 21vw, 300px);
    --ms-ease: cubic-bezier(.2, .7, .2, 1);

    /* Map the tokens onto Gradio's own variables so every built-in
       component (inputs, sliders, chatbot bubbles...) follows the theme. */
    --body-background-fill: var(--ms-bg);
    --body-text-color: var(--ms-text);
    --body-text-color-subdued: var(--ms-muted);
    --background-fill-primary: var(--ms-surface);
    --background-fill-secondary: var(--ms-surface-2);
    --border-color-primary: var(--ms-line);
    --border-color-accent: rgb(var(--ms-accent-rgb) / .45);
    --border-color-accent-subdued: rgb(var(--ms-accent-rgb) / .3);
    --color-accent: var(--ms-accent);
    --color-accent-soft: rgb(var(--ms-accent-rgb) / .12);
    --block-background-fill: var(--ms-surface);
    --block-border-color: var(--ms-line);
    --block-border-width: 1px;
    --block-radius: var(--ms-r-md);
    --block-shadow: none;
    --block-label-text-color: var(--ms-muted);
    --block-title-text-color: var(--ms-muted);
    --block-info-text-color: var(--ms-muted);
    --panel-background-fill: var(--ms-surface);
    --input-background-fill: var(--ms-surface-2);
    --input-background-fill-hover: var(--ms-surface-2);
    --input-background-fill-focus: var(--ms-surface);
    --input-border-color: var(--ms-line);
    --input-border-color-hover: var(--ms-line-strong);
    --input-border-color-focus: var(--ms-accent);
    --input-radius: var(--ms-r-sm);
    --input-shadow: none;
    --input-shadow-focus: 0 0 0 3px rgb(var(--ms-accent-rgb) / .22);
    --input-placeholder-color: var(--ms-muted);
    --button-primary-background-fill: var(--ms-accent);
    --button-primary-background-fill-hover: var(--ms-accent-strong);
    --button-primary-text-color: var(--ms-on-accent);
    --button-primary-border-color: transparent;
    --button-secondary-background-fill: var(--ms-surface);
    --button-secondary-background-fill-hover: var(--ms-surface-2);
    --button-secondary-text-color: var(--ms-text);
    --button-secondary-border-color: var(--ms-line);
    --button-large-radius: var(--ms-r-sm);
    --checkbox-background-color-selected: var(--ms-accent);
    --slider-color: var(--ms-accent);
    --shadow-drop: none;
    --chatbot-text-size: 15px;
}

.dark .gradio-container.gradio-container,
.gradio-container.gradio-container.dark {
    --ms-bg: #0a1013;
    --ms-side: #0d1519;
    --ms-surface: #121c21;
    --ms-surface-2: #172530;
    --ms-line: #1f3139;
    --ms-line-strong: #2d4652;
    --ms-text: #e6f0ed;
    --ms-muted: #8ea59f;
    --ms-accent: #2dd4bf;
    --ms-accent-strong: #5eead4;
    --ms-accent-rgb: 45 212 191;
    --ms-on-accent: #04211d;
    --ms-danger: #f87171;
    --ms-danger-rgb: 248 113 113;
    --ms-shadow: 0 1px 2px rgb(0 0 0 / .4), 0 14px 32px -16px rgb(0 0 0 / .7);
}

/* ===== 2. Page containment: nothing may escape the viewport ============= */
html, body {
    margin: 0 !important;
    max-width: 100%;
    overflow-x: hidden;
    background: #edf2f0;
}
.dark body, body.dark, html.dark { background: #0a1013; }

gradio-app { display: block; width: 100%; }

.gradio-container.gradio-container {
    width: 100% !important;
    max-width: none !important;
    min-height: 100dvh;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden;
    color: var(--ms-text);
    background: var(--ms-bg);
    font-family: var(--font);
    -webkit-font-smoothing: antialiased;
}
.gradio-container main {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
.gradio-container footer { display: none !important; }

/* Gradio wraps consecutive inputs in a bordered ".form" box; flatten it. */
#app-shell .form {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
    gap: 12px;
}
#app-shell .form > * { border-top: 0 !important; }

/* ===== 3. Shell: fixed sidebar + fluid panel, both hard-bounded ========= */
#app-shell {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 0 !important;
    width: 100% !important;
    height: 100dvh !important;
    min-height: 560px;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden;
    border: 0 !important;
    box-sizing: border-box;
}
#app-shell > * { min-width: 0 !important; box-sizing: border-box; }

#app-sidebar {
    display: flex !important;
    flex-direction: column;
    flex: 0 0 var(--ms-sidebar) !important;
    width: var(--ms-sidebar) !important;
    max-width: var(--ms-sidebar) !important;
    height: 100%;
    min-height: 0;
    gap: 12px !important;
    padding: 22px 14px 16px !important;
    overflow-x: hidden;
    overflow-y: auto;
    background: var(--ms-side) !important;
    border-right: 1px solid var(--ms-line) !important;
}
#chat-panel {
    display: flex !important;
    flex-direction: column;
    flex: 1 1 0 !important;
    width: auto !important;
    height: 100%;
    min-height: 0;
    gap: 16px !important;
    padding: 22px clamp(16px, 3vw, 44px) 22px !important;
    overflow: hidden;
    background: transparent !important;
    border: 0 !important;
}
#chat-panel > .form { flex: 0 0 auto; }

/* thin, quiet scrollbars everywhere */
#app-shell * { scrollbar-width: thin; scrollbar-color: var(--ms-line-strong) transparent; }

/* ===== 4. Sidebar ======================================================= */
#brand, #chat-heading {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
#brand h1 {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 2px;
    font: 700 22px/1.1 var(--ms-display);
    letter-spacing: -.015em;
    color: var(--ms-text);
}
#brand h1::before {                       /* dermatoscope lens mark */
    content: "";
    flex: 0 0 26px;
    height: 26px;
    border-radius: 50%;
    background:
        radial-gradient(circle, var(--ms-accent) 0 24%, transparent 26% 50%,
                        rgb(var(--ms-accent-rgb) / .28) 52% 100%);
    box-shadow: 0 0 0 2px rgb(var(--ms-accent-rgb) / .4);
}
#brand p { margin: 0; font-size: 13px; color: var(--ms-muted); }

#app-sidebar button {
    width: 100%;
    min-height: 42px;
    font-weight: 600;
    transition: transform .16s var(--ms-ease), box-shadow .2s var(--ms-ease),
                background-color .2s, border-color .2s, color .2s;
}
#new-chat { box-shadow: 0 6px 16px -8px rgb(var(--ms-accent-rgb) / .8); }
#new-chat:hover { transform: translateY(-1px); box-shadow: 0 10px 20px -8px rgb(var(--ms-accent-rgb) / .9); }
#app-sidebar button:active { transform: translateY(0) scale(.985); }

#delete-chat:hover {
    color: var(--ms-danger) !important;
    border-color: rgb(var(--ms-danger-rgb) / .5) !important;
    background: rgb(var(--ms-danger-rgb) / .08) !important;
}

/* history list takes the free height and scrolls inside itself */
#app-sidebar > .form { flex: 1 1 0; min-height: 140px; display: flex; flex-direction: column; }
#history-list {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    overflow: hidden;
}
#history-list .wrap {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    gap: 4px;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 2px;
}
#history-list label {
    position: relative;
    align-items: flex-start;
    width: 100%;
    padding: 9px 12px 9px 14px;
    border: 1px solid transparent !important;
    border-radius: var(--ms-r-sm);
    background: transparent !important;
    cursor: pointer;
    transition: background-color .18s, border-color .18s, transform .18s var(--ms-ease);
}
#history-list label:hover { background: var(--ms-surface-2) !important; transform: translateX(2px); }
#history-list label.selected {
    background: rgb(var(--ms-accent-rgb) / .12) !important;
    border-color: rgb(var(--ms-accent-rgb) / .3) !important;
}
#history-list label.selected::before {
    content: "";
    position: absolute;
    left: 0; top: 10px; bottom: 10px;
    width: 3px;
    border-radius: 3px;
    background: var(--ms-accent);
}
#history-list label span {
    display: block;
    min-width: 0;
    overflow: hidden;
    white-space: pre-line;
    text-overflow: ellipsis;
    font-size: 12.5px;
    line-height: 1.4;
    color: var(--ms-muted);
}
#history-list label span::first-line { font-size: 14px; font-weight: 600; color: var(--ms-text); }
#history-list input[type="radio"] {        /* hidden, still keyboard focusable */
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
}
#history-list label:has(input:focus-visible) { outline: 2px solid var(--ms-accent); outline-offset: 2px; }

#settings-accordion {
    flex: 0 0 auto;
    border: 1px solid var(--ms-line) !important;
    border-radius: var(--ms-r-md) !important;
    background: var(--ms-surface) !important;
}
#settings-accordion .label-wrap { padding: 12px 14px; font-weight: 600; }
#settings-accordion .form { padding: 4px 2px 8px; }

/* ===== 5. Chat panel ==================================================== */
#chat-header {
    display: flex !important;
    flex: 0 0 auto;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px 20px !important;
    padding: 0 !important;
    border: 0 !important;
}
#chat-header > * { min-width: 0 !important; flex: 0 1 auto !important; }
#chat-heading { flex: 1 1 260px !important; }
#chat-heading h2 {
    margin: 0 0 3px;
    font: 700 24px/1.15 var(--ms-display);
    letter-spacing: -.015em;
    color: var(--ms-text);
}
#chat-heading p { margin: 0; font-size: 13px; line-height: 1.5; color: var(--ms-muted); }

#clinical-toggle {
    width: auto !important;
    padding: 8px 14px 8px 12px !important;
    border: 1px solid var(--ms-line) !important;
    border-radius: 999px !important;
    background: var(--ms-surface) !important;
}
#clinical-toggle label { display: flex; align-items: center; gap: 10px; cursor: pointer; }
#clinical-toggle span { font-size: 13px; font-weight: 500; color: var(--ms-text); }
#clinical-toggle input[type="checkbox"] {   /* switch */
    appearance: none;
    -webkit-appearance: none;
    position: relative;
    flex: 0 0 36px;
    width: 36px;
    height: 20px;
    margin: 0;
    border: 0 !important;
    border-radius: 999px;
    background: var(--ms-line-strong) !important;
    background-image: none !important;
    cursor: pointer;
    transition: background-color .22s var(--ms-ease);
}
#clinical-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px; left: 2px;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 1px 3px rgb(0 0 0 / .3);
    transition: transform .24s var(--ms-ease);
}
#clinical-toggle input[type="checkbox"]:checked { background: var(--ms-accent) !important; }
#clinical-toggle input[type="checkbox"]:checked::after { transform: translateX(16px); }

#chat-header .form {                     /* radio + toggle sit side by side */
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px !important;
    width: auto !important;
    min-width: 0 !important;
    flex: 0 1 auto !important;
}
#image-type {                            /* segmented control */
    width: auto !important;
    padding: 4px !important;
    border: 1px solid var(--ms-line) !important;
    border-radius: 999px !important;
    background: var(--ms-surface) !important;
}
#image-type .wrap { flex-wrap: nowrap; gap: 2px; }
#image-type label {
    position: relative;
    padding: 6px 14px;
    border: 0 !important;
    border-radius: 999px;
    background: transparent !important;
    cursor: pointer;
    transition: background-color .2s var(--ms-ease), color .2s;
}
#image-type label span { font-size: 13px; font-weight: 500; color: var(--ms-muted); }
#image-type label.selected { background: var(--ms-accent) !important; }
#image-type label.selected span { color: var(--ms-on-accent); }
#image-type input[type="radio"] {
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
}
#image-type label:has(input:focus-visible) { outline: 2px solid var(--ms-accent); outline-offset: 2px; }

/* conversation viewport */
#conversation-view {
    position: relative;
    isolation: isolate;
    flex: 1 1 0 !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    border: 1px solid var(--ms-line) !important;
    border-radius: var(--ms-r-lg) !important;
    background: var(--ms-surface) !important;
    box-shadow: var(--ms-shadow);
}
#conversation-view > * { min-height: 0; }
#conversation-view .bubble-wrap {
    height: 100%;
    overflow-x: hidden;
    overflow-y: auto;
    padding: 18px clamp(4px, 2vw, 20px) 8px;
    background: transparent !important;
    scroll-behavior: smooth;
}
#conversation-view .message-row {
    animation: ms-rise .24s var(--ms-ease) both;
}
#conversation-view .message-row.bubble { margin: 10px 14px 4px; }
#conversation-view .message {
    max-width: 100%;
    border-radius: var(--ms-r-md) !important;
    line-height: 1.55;
    overflow-wrap: anywhere;
}
#conversation-view .user {
    background: var(--ms-accent) !important;
    border-color: transparent !important;
    border-bottom-right-radius: 5px !important;
}
#conversation-view .user, #conversation-view .user * { color: var(--ms-on-accent) !important; }
#conversation-view .bot {
    background: var(--ms-surface-2) !important;
    border-color: var(--ms-line) !important;
    border-bottom-left-radius: 5px !important;
    text-align: left !important;
}

/* generated images "develop" into focus, like a print in the tray */
#conversation-view .message-row img {
    display: block;
    width: auto;
    max-width: 100%;
    height: auto;
    max-height: min(46dvh, 420px) !important;
    margin: 2px 0 !important;
    border-radius: var(--ms-r-sm);
    object-fit: contain;
    animation: ms-develop .9s var(--ms-ease) both;
}

/* empty state: a slow scanning lens, only while no messages exist */
#conversation-view:has(.placeholder-content)::before,
#conversation-view:has(.placeholder-content)::after {
    content: "";
    position: absolute;
    z-index: 0;
    left: 50%; top: 50%;
    width: min(380px, 78%);
    aspect-ratio: 1;
    margin: 0;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
}
#conversation-view:has(.placeholder-content)::before {
    background:
        linear-gradient(var(--ms-line), var(--ms-line)) center / 1px 100% no-repeat,
        linear-gradient(var(--ms-line), var(--ms-line)) center / 100% 1px no-repeat,
        radial-gradient(circle, transparent 0 44%, var(--ms-line) 44.4% 45%,
                        transparent 45.4% 66%, var(--ms-line) 66.4% 67%,
                        transparent 67.4% 88%, var(--ms-line-strong) 88.4% 89%,
                        transparent 89.4%);
    opacity: .9;
}
#conversation-view:has(.placeholder-content)::after {
    background: conic-gradient(from 0deg, transparent 0 68%,
                               rgb(var(--ms-accent-rgb) / .22) 100%);
    -webkit-mask: radial-gradient(circle, transparent 0 22%, #000 23%);
            mask: radial-gradient(circle, transparent 0 22%, #000 23%);
    animation: ms-sweep 7s linear infinite;
}
#conversation-view .placeholder-content {
    position: relative;
    z-index: 1;
    align-items: center;
    justify-content: center;
    max-width: 280px;
    margin: 0 auto;
    text-align: center;
    color: var(--ms-muted);
    font-size: 14px;
    line-height: 1.55;
}
#conversation-view .placeholder-content strong {
    display: block;
    margin-bottom: 4px;
    font: 600 16px/1.3 var(--ms-display);
    color: var(--ms-text);
}

/* "generating" bubble */
.ms-pending { display: flex; align-items: center; gap: 14px; padding: 4px 2px; }
.ms-pending__frame {
    position: relative;
    flex: 0 0 84px;
    height: 84px;
    overflow: hidden;
    border-radius: var(--ms-r-sm);
    border: 1px solid var(--ms-line);
    background: linear-gradient(110deg, var(--ms-surface-2) 30%,
                                rgb(var(--ms-accent-rgb) / .16) 50%,
                                var(--ms-surface-2) 70%) 0 0 / 220% 100%;
    animation: ms-shimmer 1.6s linear infinite;
}
.ms-pending__scan {
    position: absolute;
    left: 0; right: 0; top: 0;
    height: 2px;
    background: var(--ms-accent);
    box-shadow: 0 0 12px 2px rgb(var(--ms-accent-rgb) / .7);
    animation: ms-scan 1.8s ease-in-out infinite;
}
.ms-pending__label { font-size: 14px; font-weight: 500; color: var(--ms-text); }
.ms-pending__label::after { content: ""; display: inline-block; width: 1.2em; text-align: left; animation: ms-dots 1.4s steps(4) infinite; }

/* ===== 6. Composer ====================================================== */
#prompt-box {
    padding: 0 !important;
    overflow: hidden;
    border: 1px solid var(--ms-line) !important;
    border-radius: var(--ms-r-lg) !important;
    background: var(--ms-surface) !important;
    box-shadow: var(--ms-shadow);
    transition: border-color .2s, box-shadow .25s var(--ms-ease);
}
#prompt-box:focus-within {
    border-color: var(--ms-accent) !important;
    box-shadow: 0 0 0 4px rgb(var(--ms-accent-rgb) / .16), var(--ms-shadow);
}
#prompt-box textarea {
    min-height: 58px !important;
    padding: 16px 18px !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 15px;
    line-height: 1.5;
}
#prompt-box .submit-button {
    border-radius: var(--ms-r-sm);
    padding: 0 18px;
    font-weight: 600;
    background: var(--ms-accent) !important;
    color: var(--ms-on-accent) !important;
    transition: transform .16s var(--ms-ease), background-color .2s, box-shadow .2s;
}
#prompt-box .submit-button:hover:not(:disabled) {
    background: var(--ms-accent-strong) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px -8px rgb(var(--ms-accent-rgb) / .9);
}
#prompt-box .submit-button:active:not(:disabled) { transform: scale(.97); }

/* ===== 7. Keyboard focus, motion, keyframes ============================= */
#app-shell button:focus-visible,
#app-shell summary:focus-visible { outline: 2px solid var(--ms-accent); outline-offset: 2px; }
::selection { background: rgb(var(--ms-accent-rgb) / .3); }

@keyframes ms-rise    { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes ms-develop { from { opacity: 0; filter: blur(16px) brightness(1.35) saturate(.5); transform: scale(.985); }
                        to   { opacity: 1; filter: none; transform: none; } }
@keyframes ms-sweep   { to { transform: translate(-50%, -50%) rotate(360deg); } }
@keyframes ms-shimmer { from { background-position: 100% 0; } to { background-position: -120% 0; } }
@keyframes ms-scan    { 0%, 100% { top: 0; } 50% { top: calc(100% - 2px); } }
@keyframes ms-dots    { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75%, 100% { content: "..."; } }

@media (prefers-reduced-motion: reduce) {
    #app-shell *, #app-shell *::before, #app-shell *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .001ms !important;
        scroll-behavior: auto !important;
    }
}

/* ===== 8. Narrow screens: sidebar stacks on top, still fully bounded ==== */
@media (max-width: 860px) {
    #app-shell { flex-direction: column !important; }
    #app-sidebar {
        flex: 0 0 auto !important;
        width: 100% !important;
        max-width: 100% !important;
        height: auto;
        max-height: 36dvh;
        padding: 14px 14px 12px !important;
        border-right: 0 !important;
        border-bottom: 1px solid var(--ms-line) !important;
    }
    #app-sidebar > .form { min-height: 96px; }
    #chat-panel {
        flex: 1 1 0 !important;
        width: 100% !important;
        padding: 14px 12px 12px !important;
    }
    #chat-heading h2 { font-size: 20px; }
}
"""


with gr.Blocks(
    title="Medsynth | Dermoscopy Image Studio",
    theme=gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=[
            gr.themes.Font("Instrument Sans"),
            gr.themes.Font("ui-sans-serif"),
            gr.themes.Font("system-ui"),
            gr.themes.Font("sans-serif"),
        ],
    ),
    css=APP_CSS,
    head=HEAD_HTML,
) as app:
    with gr.Row(elem_id="app-shell"):
        with gr.Column(scale=0, elem_id="app-sidebar"):
            gr.Markdown("# Medsynth\nDermoscopy image studio", elem_id="brand")
            new_chat = gr.Button(
                "New conversation",
                variant="primary",
                elem_id="new-chat",
            )
            history_selector = gr.Radio(
                choices=initial_choices,
                value=initial_conversation_id,
                label="Recent conversations",
                elem_id="history-list",
            )
            delete_chat = gr.Button(
                "Delete conversation",
                variant="secondary",
                elem_id="delete-chat",
            )

            with gr.Accordion(
                "Generation settings",
                open=False,
                elem_id="settings-accordion",
            ):
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=3,
                )
                steps = gr.Slider(
                    1, 50,
                    value=30,
                    step=1,
                    label="Inference steps",
                )
                guidance_scale = gr.Slider(
                    1, 15,
                    value=6.5,
                    step=0.5,
                    label="Guidance scale",
                )
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed (-1 for random)",
                )

        with gr.Column(scale=1, elem_id="chat-panel"):
            with gr.Row(elem_id="chat-header"):
                gr.Markdown(
                    "## Skin image studio\n"
                    "Describe a dermoscopic image. Generations are synthetic "
                    "and not for diagnosis.",
                    elem_id="chat-heading",
                )
                image_type = gr.Radio(
                    choices=list(IMAGE_TYPES),
                    value="Dermoscopy",
                    show_label=False,
                    label="Image type",
                    elem_id="image-type",
                )
                medical_prompt_enabled = gr.Checkbox(
                    value=True,
                    label="Clinical prompt enhancement",
                    elem_id="clinical-toggle",
                )

            chatbot = gr.Chatbot(
                value=_render_conversation(initial_conversation_id),
                type="messages",
                height=None,
                placeholder=(
                    "**Describe a lesion to begin**\n\n"
                    "Try: melanoma, malignant, back"
                ),
                show_label=False,
                elem_id="conversation-view",
                layout="bubble",
            )

            prompt = gr.Textbox(
                placeholder=(
                    "Diagnosis, category, body site, e.g. "
                    "melanoma, malignant, back  (Shift + Enter to generate)"
                ),
                label="Message",
                show_label=False,
                lines=2,
                max_lines=5,
                submit_btn="Generate",
                elem_id="prompt-box",
            )

    active_conversation = gr.State(initial_conversation_id)
    generate_inputs = [
        prompt,
        active_conversation,
        image_type,
        steps,
        guidance_scale,
        seed,
        medical_prompt_enabled,
        negative_prompt,
    ]
    generate_outputs = [
        chatbot,
        prompt,
        active_conversation,
        history_selector,
    ]

    prompt.submit(
        generate_image,
        inputs=generate_inputs,
        outputs=generate_outputs,
        show_api=False,
        show_progress="hidden",  # the in-chat "Generating" bubble replaces it
    )
    new_chat.click(
        start_conversation,
        outputs=generate_outputs,
        show_api=False,
    )
    # `.input` fires only on a real user click. `.change` also fired when the
    # code refreshed the list, which re-rendered the chat mid-generation.
    history_selector.input(
        select_conversation,
        inputs=history_selector,
        outputs=[chatbot, active_conversation],
        show_api=False,
    )
    delete_chat.click(
        remove_conversation,
        inputs=active_conversation,
        outputs=[chatbot, active_conversation, history_selector],
        show_api=False,
    )


def _warm_up():
    """Load the model in the background so the first prompt is not slow."""
    try:
        with _generation_lock:
            load_pipeline()
    except Exception as error:
        print(f"Model warm-up failed: {error}")


if __name__ == "__main__":
    if model_is_available():
        print(f"Using trained model: {MODEL_DIR}")
        Thread(target=_warm_up, daemon=True).start()
    else:
        print(
            f"\nNo trained model found at {MODEL_DIR}.\n"
            "Run `python train.py --stage all` first (see README.md).\n"
            "The app will still open; generating will explain what is missing.\n"
        )
    app.queue().launch()