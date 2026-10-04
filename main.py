import atexit
import html
import os
import random
import re
import shutil
import tempfile
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread

# This app only ever uses the model trained by train.py. Never touch the Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
from PIL import Image
from utils import chat_history, prompt_enhancer
from utils.prompt_enhancer import IMAGE_TYPES


# The fine-tuned model lives in the project folder (written by train.py) and is
# loaded from disk once per app start. Override with MEDSYNTH_MODEL_DIR.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get("MEDSYNTH_MODEL_DIR", PROJECT_DIR / "medsynth-model" / "pipeline")
)
_generation_lock = Lock()

IMAGE_TYPES = dict(IMAGE_TYPES)

# Note: no anatomy terms (face, nose, lips, hands...) here on purpose. The
# clinical datasets contain lesions on those body sites.
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, anime, sketch, "
    "3d render, CGI, digital art, artificial skin, plastic skin, "
    "waxy skin, excessively smooth skin, fake texture, "
    "unrealistic pigmentation, neon colors, oversaturated colors, "
    "extreme color grading, dramatic lighting, cinematic lighting, "
    "glowing lesion, excessive contrast, "
    "blur, motion blur, out of focus, low resolution, "
    "compression artifacts, noise, distorted anatomy, "
    "deformed body, duplicated body parts, malformed lesion, "
    "unnatural symmetry, impossible anatomy, "
    "watermark, text, logo, labels, arrows, "
    "UI, border, frame, "
    "food, landscape, scenery"
)

DEFAULT_SETTINGS = dict(chat_history.SETTINGS_DEFAULTS)
DEFAULT_SETTINGS["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT

TITLE_MAX_CHARS = 30

# Human-readable explanation of the toggle, shown in the settings panel.
ENHANCEMENT_HELP = (
    "The fine-tune was trained on short captions in one fixed grammar - "
    "`<view> of <diagnosis>, <category>, <body site>, <Fitzpatrick type>`. "
    "Enhancement parses what you wrote, normalises it into that grammar, "
    "infers anything you left out, and appends the imaging terms that view "
    "needs, all inside CLIP's hard 75-token budget. Without it your text is "
    "sent verbatim, truncated to the same budget."
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_effective_model_dir() -> Path:
    if (MODEL_DIR / "model_index.json").is_file():
        return MODEL_DIR
    fallback = PROJECT_DIR / "medsynth-model" / "pipeline"
    if (fallback / "model_index.json").is_file():
        return fallback
    fallback_model = PROJECT_DIR / "model" / "pipeline"
    if (fallback_model / "model_index.json").is_file():
        return fallback_model
    return MODEL_DIR


def model_is_available():
    return (get_effective_model_dir() / "model_index.json").is_file()


# The token counter needs the CLIP vocabulary, which lives next to the weights.
_tokenizer_dir = get_effective_model_dir() / "tokenizer"
if _tokenizer_dir.is_dir():
    prompt_enhancer.set_tokenizer_dir(_tokenizer_dir)


@lru_cache(maxsize=1)
def load_pipeline():
    """Load the fine-tuned pipeline from disk (cached for the app's lifetime)."""
    target_dir = get_effective_model_dir()
    if not (target_dir / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Trained model not found at {target_dir}. Run train.py first "
            "(see README.md), or set MEDSYNTH_MODEL_DIR to the folder that "
            "contains model_index.json."
        )

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        str(target_dir),
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

    if device == "cuda":
        try:
            pipeline = pipeline.to(device)
            pipeline.unet.to(memory_format=torch.channels_last)
            pipeline.vae.to(memory_format=torch.channels_last)
        except Exception:  # Catch CUDA OOM or device movement issues on low VRAM GPUs
            print("CUDA memory tight. Enabling attention slicing & CPU offload.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipeline.enable_attention_slicing()
            try:
                pipeline.enable_model_cpu_offload()
            except Exception:
                pipeline = pipeline.to("cpu")
                device = "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Loaded {target_dir} on CUDA GPU: {torch.cuda.get_device_name(0)} (FP16)")
    else:
        print(f"Loaded {target_dir} on CPU (slow). Install a CUDA build of PyTorch for speed.")

    return pipeline, device


# ---------------------------------------------------------------------------
# Transient image files
# ---------------------------------------------------------------------------

# Images live in the database as BLOBs. Gradio needs a real path to serve them,
# so each stored message is unwrapped once into this scratch directory and the
# path is reused for every later re-render instead of piling up temp files.
_SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="medsynth-chat-"))
_image_paths: dict[int, str] = {}
atexit.register(shutil.rmtree, _SCRATCH_DIR, True)


def _image_file(message_id: int, blob: bytes) -> str:
    cached = _image_paths.get(message_id)
    if cached:
        return cached
    path = _SCRATCH_DIR / f"image-{message_id}.png"
    with open(path, "wb") as handle:
        handle.write(blob)
    _image_paths[message_id] = str(path)
    return str(path)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _short_title(title):
    title = " ".join((title or "Untitled").split())
    if len(title) > TITLE_MAX_CHARS:
        title = title[: TITLE_MAX_CHARS - 1].rstrip() + "…"
    return title or "Untitled"


def _relative_stamp(value):
    """'2026-09-29 14:27:03' -> 'now' / '14m' / '3h' / '2d' / '12 Oct'."""
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return ""

    delta = datetime.now() - moment
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return moment.strftime("%d %b")
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    if seconds < 86400 * 7:
        return f"{seconds // 86400}d"
    return moment.strftime("%d %b")


def _activity_label(conversation):
    images = conversation.get("image_count") or 0
    messages = conversation.get("message_count") or 0
    if not messages:
        return "empty"
    parts = []
    if images:
        parts.append(f"{images} image" + ("s" if images != 1 else ""))
    parts.append(f"{messages} message" + ("s" if messages != 1 else ""))
    return " · ".join(parts)


def _last_prompt(conversation_id):
    """Most recent user prompt in a thread, used to pre-fill the token meter."""
    for message in reversed(chat_history.get_messages(conversation_id)):
        if message["role"] == "user":
            return str(message["content"] or "")
    return ""


def _conversation_choices(query=""):
    # Two-line labels: the title on the first line, activity + time on the
    # second. The CSS uses `white-space: pre-line` so the newline is honoured.
    return [
        (
            f"{_short_title(conversation['title'])}\n"
            f"{_activity_label(conversation)} · "
            f"{_relative_stamp(conversation['updated_at'])}",
            conversation["id"],
        )
        for conversation in chat_history.list_conversations(query)
    ]


# ---------------------------------------------------------------------------
# Transcript rendering
# ---------------------------------------------------------------------------

PENDING_HTML = """
<div class="ms-pending" role="status" aria-live="polite">
  <div class="ms-pending__frame"><div class="ms-pending__scan"></div></div>
  <div class="ms-pending__body">
    <span class="ms-pending__label">Generating image</span>
    <span class="ms-pending__hint">Diffusion sampling - this can take a
      moment on a laptop GPU</span>
  </div>
</div>
"""


def _error_html(message):
    return (
        '<div class="ms-error" role="alert">'
        '<span class="ms-error__icon" aria-hidden="true">!</span>'
        f'<span class="ms-error__text">{html.escape(str(message))}</span>'
        "</div>"
    )


def _render_conversation(conversation_id, pending=False):
    rendered = []

    for message in chat_history.get_messages(conversation_id):
        content = str(message["content"] or "")
        is_error = content.startswith("Generation failed:")

        rendered.append(
            {
                "role": message["role"],
                "content": _error_html(content) if is_error else content,
            }
        )

        if message["image"] is None:
            continue

        try:
            # Re-validate through PIL so a truncated BLOB cannot poison Gradio.
            image = Image.open(BytesIO(message["image"])).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            path = _image_file(message["id"], buffer.getvalue())
            rendered.append(
                {
                    "role": "assistant",
                    "content": {
                        "path": path,
                        "alt_text": _plain_text(content)
                        or "Generated synthetic skin image",
                    },
                }
            )
        except Exception as error:
            rendered.append(
                {
                    "role": "assistant",
                    "content": _error_html(f"Unable to display image: {error}"),
                }
            )

    if pending:
        rendered.append({"role": "assistant", "content": PENDING_HTML})

    return rendered


# ---------------------------------------------------------------------------
# Sidebar state
# ---------------------------------------------------------------------------

def _history_update(selected_id=None, query=""):
    choices = _conversation_choices(query)
    conversation_ids = {value for _, value in choices}
    if selected_id not in conversation_ids:
        selected_id = choices[0][1] if choices else None
    return gr.update(choices=choices, value=selected_id), selected_id


def _history_empty_html(query, total):
    if total == 0:
        return (
            '<div class="ms-empty">'
            "<strong>No conversations yet</strong>"
            "<span>Start one with New conversation, then describe a lesion."
            "</span></div>"
        )
    if query.strip():
        return (
            '<div class="ms-empty">'
            f"<strong>No match for “{html.escape(query.strip())}”</strong>"
            "<span>Search looks at titles and everything said inside.</span>"
            "</div>"
        )
    return ""


def _history_sidebar(selected_id=None, query=""):
    update, selected_id = _history_update(selected_id, query)
    total = len(chat_history.list_conversations())
    return update, _history_empty_html(query, total), selected_id


def _settings_updates(settings):
    return (
        settings["image_type"],
        bool(settings["clinical"]),
        int(settings["steps"]),
        float(settings["guidance"]),
        int(settings["seed"]),
        settings["negative_prompt"] or DEFAULT_NEGATIVE_PROMPT,
    )


# ---------------------------------------------------------------------------
# Prompt context bar
# ---------------------------------------------------------------------------

def _chip(label, value, tone=""):
    tone = f" ms-chip--{tone}" if tone else ""
    return (
        f'<span class="ms-chip{tone}">'
        f'<span class="ms-chip__key">{html.escape(label)}</span>'
        f'<span class="ms-chip__val">{html.escape(str(value))}</span>'
        "</span>"
    )


def _context_html(prompt, image_type, clinical, steps, guidance, seed):
    """The strip above the composer: active settings + a live token meter."""
    plan = prompt_enhancer.build_plan(prompt, clinical, image_type)

    meter_tone = (
        "warn" if plan.truncated or plan.tokens >= plan.limit - 5 else ""
    )
    meter_class = f"ms-meter ms-meter--{meter_tone}" if meter_tone else "ms-meter"

    chips = [
        _chip("view", image_type),
        _chip("steps", int(steps)),
        _chip("cfg", f"{float(guidance):g}"),
        _chip("seed", "random" if seed is None or int(seed) < 0 else int(seed)),
        _chip(
            "clinical",
            "on" if clinical else "off",
            "on" if clinical else "off",
        ),
    ]

    fill = round(plan.usage * 100)
    meter = (
        f'<div class="{meter_class}" role="img" '
        f'aria-label="Prompt uses {plan.tokens} of {plan.limit} CLIP tokens">'
        f'<span class="ms-meter__track">'
        f'<span class="ms-meter__fill" style="width:{fill}%"></span></span>'
        f'<span class="ms-meter__text">{plan.tokens}/{plan.limit} tokens</span>'
        "</div>"
    )

    notes = []
    if plan.truncated:
        notes.append(
            "<li>Trimmed to fit CLIP's 77-token ceiling - the tail was "
            "shortened.</li>"
        )
    if not plan.text and (prompt or "").strip():
        notes.append("<li>Nothing left to send; describe the lesion again.</li>")

    breakdown = ""
    if notes or plan.text:
        rows = "".join(notes)
        if plan.text:
            rows += (
                f'<li class="ms-note__text"><code>{html.escape(plan.text)}</code></li>'
            )
        breakdown = f'<ul class="ms-note">{rows}</ul>'

    return (
        f'<div class="ms-context">'
        f'<div class="ms-chips">{"".join(chips)}</div>'
        f"{meter}"
        f'<details class="ms-details"><summary>How the prompt is built</summary>'
        f"{breakdown}</details>"
        f"</div>"
    )


def sync_context(prompt, image_type, clinical, steps, guidance, seed):
    return _context_html(prompt, image_type, clinical, steps, guidance, seed)


# ---------------------------------------------------------------------------
# Settings actions
# ---------------------------------------------------------------------------

def persist_settings(
    conversation_id, image_type, clinical, steps, guidance, seed,
    negative_prompt, prompt,
):
    """Store the current controls on this conversation, and echo the context bar."""
    if conversation_id:
        chat_history.save_settings(
            conversation_id,
            image_type=image_type,
            clinical=bool(clinical),
            steps=int(steps),
            guidance=float(guidance),
            seed=-1 if seed is None else int(seed),
            negative_prompt=negative_prompt or "",
        )
    return sync_context(prompt, image_type, clinical, steps, guidance, seed)


def reset_settings(conversation_id, prompt=""):
    settings = chat_history.reset_settings(conversation_id)
    settings["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
    updates = _settings_updates(settings)
    return (*updates, sync_context(prompt, *updates[:5]))


def randomize_seed():
    return random.randint(0, 2_147_483_647)


# ---------------------------------------------------------------------------
# Conversation actions
# ---------------------------------------------------------------------------

def start_conversation(query=""):
    """Create a thread and clear the search filter so it is actually visible.

    Without clearing it, a new thread would fall outside the active filter and
    the sidebar would highlight some other conversation than the one opened.
    """
    conversation_id = chat_history.create_conversation(
        {
            "image_type": DEFAULT_SETTINGS["image_type"],
            "clinical": DEFAULT_SETTINGS["clinical"],
            "steps": DEFAULT_SETTINGS["steps"],
            "guidance": DEFAULT_SETTINGS["guidance"],
            "seed": DEFAULT_SETTINGS["seed"],
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        }
    )
    history_update, empty_html, _ = _history_sidebar(conversation_id)
    updates = _settings_updates(chat_history.get_settings(conversation_id))
    return (
        [],
        conversation_id,
        history_update,
        empty_html,
        *updates,
        sync_context("", *updates[:5]),
        gr.update(value="Delete conversation", elem_classes=[]),
        None,
        "",
    )


def select_conversation(conversation_id, query=""):
    settings = chat_history.get_settings(conversation_id)
    updates = _settings_updates(settings)
    return (
        _render_conversation(conversation_id),
        conversation_id,
        *_settings_updates(settings),
        sync_context(_last_prompt(conversation_id), *updates[:5]),
        gr.update(value="Delete conversation", elem_classes=[]),
        None,
    )


def remove_conversation(conversation_id, query=""):
    chat_history.delete_conversation(conversation_id)
    if not chat_history.list_conversations():
        chat_history.create_conversation()

    history_update, empty_html, selected_id = _history_sidebar(None, query)
    settings = chat_history.get_settings(selected_id)
    updates = _settings_updates(settings)
    return (
        _render_conversation(selected_id),
        selected_id,
        history_update,
        empty_html,
        *updates,
        sync_context(_last_prompt(selected_id), *updates[:5]),
        gr.update(value="Delete conversation", elem_classes=[]),
    )


def clear_conversation(conversation_id, query=""):
    """Empty the thread but keep it - and its settings - in the sidebar."""
    chat_history.clear_messages(conversation_id)
    history_update, empty_html, _ = _history_sidebar(conversation_id, query)
    return [], history_update, empty_html, None


def filter_history(query, conversation_id):
    history_update, empty_html, selected_id = _history_sidebar(conversation_id, query)
    return history_update, empty_html, selected_id


def arm_delete(conversation_id, pending_id):
    """First click arms, second click deletes. Never a one-click data loss.

    Gradio requires one return value per declared output, so the arming branch
    pads with `gr.skip()` to leave the transcript and settings untouched.
    """
    if not conversation_id:
        return (
            *([gr.skip()] * 11),
            gr.update(value="Nothing to delete", elem_classes=["ms-btn--armed"]),
            None,
        )

    if pending_id != conversation_id:
        return (
            *([gr.skip()] * 11),
            gr.update(value="Confirm delete", elem_classes=["ms-btn--armed"]),
            conversation_id,
        )

    return (*remove_conversation(conversation_id), None)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _metadata_parts(plan, image_type, steps, guidance, used_seed):
    """The one-line summary stored with an image: (caption, facts)."""
    caption = plan.as_caption() if plan.enhanced else (plan.text or "custom prompt")
    facts = [
        f"{int(steps)} steps",
        f"CFG {float(guidance):g}",
        f"seed {used_seed}",
        f"{plan.tokens} tokens",
    ]
    if not plan.enhanced:
        facts.insert(0, "verbatim")
    elif plan.truncated:
        facts.insert(0, "trimmed")
    return caption, facts


def _metadata_caption(plan, image_type, steps, guidance, used_seed):
    caption, facts = _metadata_parts(plan, image_type, steps, guidance, used_seed)
    return (
        '<div class="ms-cap">'
        f'<b>{html.escape(image_type)}</b> · {html.escape(caption)}<br>'
        f'<span class="ms-cap__meta">{html.escape(" · ".join(facts))}</span>'
        "</div>"
    )


def _plain_text(markup):
    """Flatten stored caption markup for screen readers and alt text."""
    text = re.sub(r"<br\s*/?>", ". ", str(markup or ""))
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).replace(" · ", ", ").replace("..", ".").strip()


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

    plan = prompt_enhancer.build_plan(
        prompt, medical_prompt_enabled, image_type,
    )

    if not prompt:
        yield (
            _render_conversation(conversation_id),
            "",
            conversation_id,
            gr.update(),
            gr.update(),
            gr.update(),
            sync_context("", image_type, medical_prompt_enabled,
                        steps, guidance_scale, seed),
        )
        return

    if not conversation_id:
        conversation_id = chat_history.create_conversation()

    chat_history.add_message(conversation_id, "user", prompt)
    chat_history.save_settings(
        conversation_id,
        image_type=image_type,
        clinical=bool(medical_prompt_enabled),
        steps=int(steps),
        guidance=float(guidance_scale),
        seed=-1 if seed is None else int(seed),
        negative_prompt=negative_prompt or "",
    )

    history_update, empty_html, _ = _history_sidebar(conversation_id)
    yield (
        _render_conversation(conversation_id, pending=True),
        "",
        conversation_id,
        history_update,
        empty_html,
        gr.update(),
        sync_context(prompt, image_type, medical_prompt_enabled,
                     steps, guidance_scale, seed),
    )

    used_seed = -1 if seed is None else int(seed)

    try:
        import torch

        with _generation_lock:
            pipeline, device = load_pipeline()

            generator = None
            if used_seed >= 0:
                generator = torch.Generator(device=device).manual_seed(used_seed)
            else:
                # Report the seed that was actually used so a good result can
                # be reproduced by pasting it back into the seed field.
                used_seed = random.randint(0, 2_147_483_647)
                generator = torch.Generator(device=device).manual_seed(used_seed)

            negative = (negative_prompt or "").strip() or None
            image = pipeline(
                prompt=plan.text,
                negative_prompt=negative,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0]

        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        chat_history.add_message(
            conversation_id,
            "assistant",
            _metadata_caption(
                plan, image_type, steps, guidance_scale, used_seed,
            ),
            image=image_buffer.getvalue(),
            image_mime="image/png",
        )
    except Exception as error:
        chat_history.add_message(
            conversation_id,
            "assistant",
            f"Generation failed: {error}",
        )

    history_update, empty_html, _ = _history_sidebar(conversation_id)
    yield (
        _render_conversation(conversation_id),
        "",
        conversation_id,
        history_update,
        empty_html,
        gr.update(),
        sync_context("", image_type, medical_prompt_enabled,
                     steps, guidance_scale, seed),
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

initial_choices = _conversation_choices()
if not initial_choices:
    chat_history.create_conversation()
    initial_choices = _conversation_choices()
initial_conversation_id = initial_choices[0][1]
initial_settings = chat_history.get_settings(initial_conversation_id)
if not initial_settings["negative_prompt"]:
    initial_settings["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
initial_updates = _settings_updates(initial_settings)


_model_state: dict[str, str] = {"state": "loading", "detail": ""}


def _status_pill(tone, label):
    return (
        f'<div class="ms-status ms-status--{tone}" role="status" aria-live="polite">'
        '<span class="ms-status__dot" aria-hidden="true"></span>'
        f"{label}</div>"
    )


def model_status_html():
    """A small pill in the sidebar so the model state is never a mystery."""
    state, detail = _model_state["state"], _model_state["detail"]

    if not model_is_available():
        return _status_pill(
            "error", "No trained model &mdash; run <code>train.py</code> first",
        )
    if state == "error":
        return _status_pill("error", f"Model failed to load &mdash; {detail}")
    if state == "loading":
        return _status_pill("loading", "Loading model&hellip;")
    return _status_pill("ready", f"Model ready &middot; {detail}")


def model_status_tick():
    """Timer callback: only speaks up while the model is still loading."""
    if _model_state["state"] == "loading":
        return model_status_html()
    return gr.skip()


def _describe_device():
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
            return f"{name} &middot; FP16"
        return "CPU &middot; slow"
    except Exception:
        return "device unknown"


# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------

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
    --ms-warn: #a16207;
    --ms-warn-rgb: 161 98 7;
    --ms-shadow: 0 1px 2px rgb(16 32 28 / .05), 0 10px 28px -14px rgb(16 32 28 / .22);
    --ms-shadow-sm: 0 1px 2px rgb(16 32 28 / .06);

    --ms-r-xs: 7px;
    --ms-r-sm: 10px;
    --ms-r-md: 14px;
    --ms-r-lg: 20px;
    --ms-sidebar: clamp(252px, 21vw, 304px);
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
    --ms-warn: #fbbf24;
    --ms-warn-rgb: 251 191 36;
    --ms-shadow: 0 1px 2px rgb(0 0 0 / .4), 0 14px 32px -16px rgb(0 0 0 / .7);
    --ms-shadow-sm: 0 1px 2px rgb(0 0 0 / .35);
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

/* A generic section label: small caps, muted, quiet */
#app-shell .ms-label {
    margin: 0;
    font: 600 11px/1 var(--font);
    letter-spacing: .09em;
    text-transform: uppercase;
    color: var(--ms-muted);
}
#app-shell .ms-hint {
    margin: 0;
    font-size: 11.5px;
    line-height: 1.45;
    color: var(--ms-muted);
}
#app-shell .ms-hint code, .ms-status code, .ms-note code {
    padding: 1px 5px;
    border-radius: 5px;
    background: var(--ms-surface-2);
    border: 1px solid var(--ms-line);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: .92em;
    overflow-wrap: anywhere;
}

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
    gap: 10px !important;
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
    gap: 14px !important;
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
    margin: 0 0 3px;
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

/* model readiness pill */
.ms-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border: 1px solid var(--ms-line);
    border-radius: 999px;
    background: var(--ms-surface);
    font-size: 11.5px;
    line-height: 1.3;
    color: var(--ms-muted);
}
.ms-status__dot {
    flex: 0 0 7px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ms-line-strong);
}
.ms-status--ready .ms-status__dot {
    background: var(--ms-accent);
    box-shadow: 0 0 0 3px rgb(var(--ms-accent-rgb) / .18);
}
.ms-status--loading .ms-status__dot {
    background: var(--ms-warn);
    box-shadow: 0 0 0 3px rgb(var(--ms-warn-rgb) / .2);
    animation: ms-pulse 1.3s ease-in-out infinite;
}
.ms-status--error {
    border-color: rgb(var(--ms-danger-rgb) / .4);
    background: rgb(var(--ms-danger-rgb) / .07);
    color: var(--ms-danger);
}
.ms-status--error .ms-status__dot { background: var(--ms-danger); }

#app-sidebar button {
    width: 100%;
    min-height: 40px;
    font-weight: 600;
    transition: transform .16s var(--ms-ease), box-shadow .2s var(--ms-ease),
                background-color .2s, border-color .2s, color .2s;
}
#new-chat { box-shadow: 0 6px 16px -8px rgb(var(--ms-accent-rgb) / .8); }
#new-chat:hover { transform: translateY(-1px); box-shadow: 0 10px 20px -8px rgb(var(--ms-accent-rgb) / .9); }
#new-chat:active { transform: translateY(0) scale(.985); }

#delete-chat:hover {
    color: var(--ms-danger) !important;
    border-color: rgb(var(--ms-danger-rgb) / .5) !important;
    background: rgb(var(--ms-danger-rgb) / .08) !important;
}
.ms-btn--armed, #delete-chat.ms-btn--armed {
    color: var(--ms-on-accent) !important;
    background: var(--ms-danger) !important;
    border-color: transparent !important;
    animation: ms-throb 1.6s var(--ms-ease) infinite;
}

/* history block takes the free height and scrolls inside itself */
#app-sidebar > #history-form {
    flex: 1 1 0;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    gap: 8px !important;
}
#history-search { flex: 0 0 auto; }
#history-search input {
    font-size: 13px !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}
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
    gap: 3px;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 3px;
}
#history-list label {
    position: relative;
    align-items: flex-start;
    width: 100%;
    padding: 8px 12px 9px 14px;
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
    left: 0; top: 9px; bottom: 9px;
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
    font-size: 12px;
    line-height: 1.45;
    color: var(--ms-muted);
}
#history-list label span::first-line {
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: -.005em;
    color: var(--ms-text);
}
#history-list input[type="radio"] {        /* hidden, still keyboard focusable */
    position: absolute; opacity: 0; width: 0; height: 0; pointer-events: none;
}
#history-list label:has(input:focus-visible) { outline: 2px solid var(--ms-accent); outline-offset: 2px; }

/* sidebar empty / no-match state */
#history-empty:empty, #model-status:empty { display: none; }
.ms-empty {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 14px 12px;
    border: 1px dashed var(--ms-line-strong);
    border-radius: var(--ms-r-sm);
    text-align: center;
}
.ms-empty strong { font: 600 13px/1.35 var(--ms-display); color: var(--ms-text); }
.ms-empty span { font-size: 11.5px; line-height: 1.45; color: var(--ms-muted); }

#settings-accordion {
    flex: 0 0 auto;
    border: 1px solid var(--ms-line) !important;
    border-radius: var(--ms-r-md) !important;
    background: var(--ms-surface) !important;
}
#settings-accordion .label-wrap {
    padding: 11px 14px;
    font-weight: 600;
    font-size: 13.5px;
}
#settings-accordion .label-wrap svg { color: var(--ms-muted); }
#settings-accordion .form { padding: 2px 2px 10px; gap: 14px !important; }
#settings-accordion .ms-note-box {
    margin: 0;
    padding: 10px 11px;
    border-radius: var(--ms-r-xs);
    background: var(--ms-surface-2);
    border: 1px solid var(--ms-line);
    font-size: 11.5px;
    line-height: 1.5;
    color: var(--ms-muted);
}
#seed-row { align-items: flex-end !important; gap: 8px !important; }
#seed-row > #seed-field { flex: 1 1 auto !important; min-width: 0 !important; }
#seed-row > #seed-shuffle { flex: 0 0 auto !important; width: auto !important; }
#seed-shuffle { min-height: 38px !important; padding: 0 12px !important; white-space: nowrap; }

/* ===== 5. Chat panel ==================================================== */
#chat-header {
    display: flex !important;
    flex: 0 0 auto;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px 20px !important;
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

#header-controls {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px !important;
    width: auto !important;
    min-width: 0 !important;
    flex: 0 1 auto !important;
}
#header-controls .form { display: contents !important; }

#clinical-toggle {
    width: auto !important;
    padding: 7px 14px 7px 12px !important;
    border: 1px solid var(--ms-line) !important;
    border-radius: 999px !important;
    background: var(--ms-surface) !important;
    transition: border-color .2s, background-color .2s, box-shadow .2s;
}
#clinical-toggle.is-on {
    border-color: rgb(var(--ms-accent-rgb) / .45) !important;
    background: rgb(var(--ms-accent-rgb) / .08) !important;
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
#clinical-toggle label:has(input:focus-visible) { outline: 2px solid var(--ms-accent); outline-offset: 3px; border-radius: 999px; }

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
#image-type label span { font-size: 13px; font-weight: 500; color: var(--ms-muted); white-space: nowrap; }
#image-type label:hover:not(.selected) { background: var(--ms-surface-2) !important; }
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
    padding: 20px clamp(4px, 2vw, 20px) 12px;
    background: transparent !important;
    scroll-behavior: smooth;
    scroll-padding-bottom: 16px;
}
#conversation-view .message-row { animation: ms-rise .24s var(--ms-ease) both; }
#conversation-view .message-row.bubble { margin: 10px 14px 4px; }
/* caption + image + error blocks sit tight together, no bubble around them */
#conversation-view .bot:has(.ms-cap) + .message-row.bubble { margin-top: 0 !important; }
#conversation-view .message {
    max-width: min(100%, 620px);
    border-radius: var(--ms-r-md) !important;
    line-height: 1.55;
    overflow-wrap: anywhere;
}
#conversation-view .message p { margin: 0 0 .45em; }
#conversation-view .message p:last-child { margin-bottom: 0; }
#conversation-view .message-row.user-row { display: flex; justify-content: flex-end; }
#conversation-view .message-row.bot-row { display: flex; justify-content: flex-start; }
#conversation-view .user {
    background: var(--ms-accent) !important;
    border-color: transparent !important;
    border-bottom-right-radius: 5px !important;
    max-width: min(86%, 560px);
    box-shadow: 0 4px 14px -10px rgb(var(--ms-accent-rgb) / .9);
}
#conversation-view .user, #conversation-view .user * { color: var(--ms-on-accent) !important; }
#conversation-view .bot {
    background: var(--ms-surface-2) !important;
    border-color: var(--ms-line) !important;
    border-bottom-left-radius: 5px !important;
    text-align: left !important;
}
/* caption + image + error blocks are chrome, not prose: no bubble around them */
#conversation-view .bot:has(.ms-cap),
#conversation-view .bot:has(.ms-error),
#conversation-view .bot:has(.ms-pending),
#conversation-view .bot:has(img) {
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
    padding: 2px !important;
    max-width: 100%;
}
/* the caption introduces the image right below it */
#conversation-view .bot:has(.ms-cap) { padding-bottom: 4px !important; }
#conversation-view .bot:has(img) { padding-top: 0 !important; }

.ms-cap {
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--ms-muted);
    padding-left: 2px;
}
.ms-cap b { color: var(--ms-text); font-weight: 600; }
.ms-cap__meta { font-size: 11.5px; color: var(--ms-muted); opacity: .85; }

.ms-error {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    max-width: 100%;
    padding: 10px 13px;
    border: 1px solid rgb(var(--ms-danger-rgb) / .35);
    border-radius: var(--ms-r-sm);
    background: rgb(var(--ms-danger-rgb) / .08);
    font-size: 13px;
    line-height: 1.5;
    color: var(--ms-danger);
}
.ms-error__icon {
    flex: 0 0 18px;
    width: 18px;
    height: 18px;
    margin-top: 1px;
    border-radius: 50%;
    background: var(--ms-danger);
    color: #fff;
    font: 700 12px/18px var(--font);
    text-align: center;
}
.ms-error__text { min-width: 0; overflow-wrap: anywhere; }

/* generated images "develop" into focus, like a print in the tray */
#conversation-view .message-row img {
    display: block;
    width: auto;
    max-width: 100%;
    height: auto;
    max-height: min(46dvh, 420px) !important;
    margin: 2px 0 !important;
    border-radius: var(--ms-r-sm);
    border: 1px solid var(--ms-line);
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
    width: min(360px, 74%);
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
    max-width: 300px;
    margin: 0 auto;
    text-align: center;
    color: var(--ms-muted);
    font-size: 14px;
    line-height: 1.55;
}
#conversation-view .placeholder-content strong {
    display: block;
    margin-bottom: 6px;
    font: 600 17px/1.3 var(--ms-display);
    letter-spacing: -.01em;
    color: var(--ms-text);
}

/* "generating" bubble */
.ms-pending { display: flex; align-items: center; gap: 14px; padding: 4px 2px; }
.ms-pending__frame {
    position: relative;
    flex: 0 0 78px;
    height: 78px;
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
.ms-pending__body { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.ms-pending__label { font-size: 14px; font-weight: 600; color: var(--ms-text); }
.ms-pending__label::after { content: ""; display: inline-block; width: 1.2em; text-align: left; animation: ms-dots 1.4s steps(4) infinite; }
.ms-pending__hint { font-size: 12px; line-height: 1.45; color: var(--ms-muted); }

/* ===== 6. Composer ====================================================== */
#composer { flex: 0 0 auto; display: flex !important; flex-direction: column; gap: 8px !important; }

/* context strip: active settings chips + live token meter + prompt preview */
.ms-context {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 11px 13px;
    border: 1px solid var(--ms-line);
    border-radius: var(--ms-r-md);
    background: var(--ms-surface);
    box-shadow: var(--ms-shadow-sm);
}
.ms-chips { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.ms-chip {
    display: inline-flex;
    align-items: baseline;
    gap: 5px;
    padding: 3px 9px;
    border: 1px solid var(--ms-line);
    border-radius: 999px;
    background: var(--ms-surface-2);
    font-size: 11.5px;
    line-height: 1.5;
    white-space: nowrap;
}
.ms-chip__key { color: var(--ms-muted); text-transform: uppercase; letter-spacing: .06em; font-size: 10px; }
.ms-chip__val { color: var(--ms-text); font-weight: 600; }
.ms-chip--on {
    border-color: rgb(var(--ms-accent-rgb) / .4);
    background: rgb(var(--ms-accent-rgb) / .12);
}
.ms-chip--on .ms-chip__val { color: var(--ms-accent); }
.ms-chip--off { opacity: .72; }

.ms-meter { display: flex; align-items: center; gap: 9px; }
.ms-meter__track {
    flex: 1 1 auto;
    height: 5px;
    border-radius: 999px;
    background: var(--ms-surface-2);
    border: 1px solid var(--ms-line);
    overflow: hidden;
}
.ms-meter__fill {
    display: block;
    height: 100%;
    border-radius: 999px;
    background: var(--ms-accent);
    transition: width .28s var(--ms-ease), background-color .2s;
}
.ms-meter--warn .ms-meter__fill { background: var(--ms-warn); }
.ms-meter__text {
    flex: 0 0 auto;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    color: var(--ms-muted);
}
.ms-meter--warn .ms-meter__text { color: var(--ms-warn); font-weight: 600; }

.ms-details > summary {
    cursor: pointer;
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: var(--ms-muted);
    list-style: none;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: color .18s;
}
.ms-details > summary::-webkit-details-marker { display: none; }
.ms-details > summary::before {
    content: "";
    width: 0; height: 0;
    border: 4px solid transparent;
    border-left-color: currentColor;
    transition: transform .2s var(--ms-ease);
}
.ms-details[open] > summary::before { transform: rotate(90deg); }
.ms-details > summary:hover { color: var(--ms-text); }
.ms-details > summary:focus-visible { outline: 2px solid var(--ms-accent); outline-offset: 3px; border-radius: 4px; }
.ms-note {
    margin: 8px 0 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 12px;
    line-height: 1.5;
    color: var(--ms-muted);
}
.ms-note li { margin: 0; }
.ms-note li::before {
    content: "";
    display: inline-block;
    width: 5px; height: 5px;
    margin-right: 7px;
    border-radius: 50%;
    background: var(--ms-warn);
    vertical-align: middle;
}
.ms-note__text::before { display: none; }
.ms-note__text code { display: inline-block; line-height: 1.5; }

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
    min-height: 56px !important;
    padding: 15px 18px 6px !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 15px;
    line-height: 1.5;
}
#prompt-box .submit-button {
    border-radius: var(--ms-r-sm);
    margin: 0 12px 12px 0;
    padding: 8px 20px;
    font-weight: 600;
    background: var(--ms-accent) !important;
    color: var(--ms-on-accent) !important;
    transition: transform .16s var(--ms-ease), background-color .2s, box-shadow .2s, opacity .2s;
}
#prompt-box .submit-button:hover:not(:disabled) {
    background: var(--ms-accent-strong) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px -8px rgb(var(--ms-accent-rgb) / .9);
}
#prompt-box .submit-button:active:not(:disabled) { transform: scale(.97); }

/* ===== 7. Keyboard focus, motion, keyframes ============================= */
#app-shell button:focus-visible,
#app-shell summary:focus-visible,
#app-shell [role="radio"]:focus-visible { outline: 2px solid var(--ms-accent); outline-offset: 2px; }
::selection { background: rgb(var(--ms-accent-rgb) / .3); }

@keyframes ms-rise    { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes ms-develop { from { opacity: 0; filter: blur(16px) brightness(1.35) saturate(.5); transform: scale(.985); }
                        to   { opacity: 1; filter: none; transform: none; } }
@keyframes ms-sweep   { to { transform: translate(-50%, -50%) rotate(360deg); } }
@keyframes ms-shimmer { from { background-position: 100% 0; } to { background-position: -120% 0; } }
@keyframes ms-scan    { 0%, 100% { top: 0; } 50% { top: calc(100% - 2px); } }
@keyframes ms-dots    { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75%, 100% { content: "..."; } }
@keyframes ms-throb   { 0%, 100% { box-shadow: 0 0 0 0 rgb(var(--ms-danger-rgb) / .45); }
                        60% { box-shadow: 0 0 0 7px rgb(var(--ms-danger-rgb) / 0); } }
@keyframes ms-pulse   { 0%, 100% { opacity: 1; transform: scale(1); }
                        50% { opacity: .45; transform: scale(.82); } }

@media (prefers-reduced-motion: reduce) {
    #app-shell *, #app-shell *::before, #app-shell *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .001ms !important;
        scroll-behavior: auto !important;
    }
}

/* ===== 8. Narrow screens: sidebar stacks on top, still fully bounded ==== */
@media (max-width: 900px) {
    #app-shell { flex-direction: column !important; }
    #app-sidebar {
        flex: 0 0 auto !important;
        width: 100% !important;
        max-width: 100% !important;
        height: auto;
        max-height: 42dvh;
        padding: 14px 14px 12px !important;
        border-right: 0 !important;
        border-bottom: 1px solid var(--ms-line) !important;
    }
    #app-sidebar > #history-form { min-height: 110px; }
    #chat-panel {
        flex: 1 1 0 !important;
        width: 100% !important;
        padding: 14px 12px 12px !important;
    }
    #chat-heading h2 { font-size: 20px; }
    #header-controls { width: 100% !important; }
}

@media (max-width: 560px) {
    #brand h1 { font-size: 19px; }
    #chat-heading h2 { font-size: 18px; }
    #image-type label { padding: 6px 10px; }
    #image-type label span { font-size: 12px; }
    #conversation-view .message-row.bubble { margin: 8px 4px 4px; }
    #conversation-view .message { max-width: 100%; }
    #conversation-view .message-row img { max-height: min(38dvh, 300px) !important; }
    #prompt-box textarea { padding: 13px 14px 4px !important; }
    #prompt-box .submit-button { margin: 0 10px 10px 0; padding: 7px 16px; }
    .ms-context { padding: 10px 11px; }
    .ms-pending { gap: 10px; }
    .ms-pending__frame { flex: 0 0 60px; height: 60px; }
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
        # ---------------- sidebar ----------------
        with gr.Column(scale=0, elem_id="app-sidebar"):
            gr.Markdown("# Medsynth\nDermoscopy image studio", elem_id="brand")
            model_status = gr.HTML(model_status_html(), elem_id="model-status")

            new_chat = gr.Button(
                "New conversation",
                variant="primary",
                elem_id="new-chat",
            )

            with gr.Column(elem_id="history-form"):
                gr.HTML(
                    '<p class="ms-label">Conversations</p>', elem_id="history-label",
                )
                history_search = gr.Textbox(
                    placeholder="Search titles and prompts",
                    show_label=False,
                    elem_id="history-search",
                    container=False,
                )
                history_empty = gr.HTML("", elem_id="history-empty")
                history_selector = gr.Radio(
                    choices=initial_choices,
                    value=initial_conversation_id,
                    show_label=False,
                    elem_id="history-list",
                )

            with gr.Row():
                clear_chat = gr.Button(
                    "Clear chat", variant="secondary", elem_id="clear-chat",
                    scale=1,
                )
                delete_chat = gr.Button(
                    "Delete conversation", variant="secondary",
                    elem_id="delete-chat", scale=1,
                )

            with gr.Accordion(
                "Generation settings",
                open=False,
                elem_id="settings-accordion",
            ):
                steps = gr.Slider(
                    1, 50,
                    value=int(initial_updates[2]),
                    step=1,
                    label="Inference steps",
                    info="Diffusion passes. More is cleaner but slower; 30 suits a laptop GPU.",
                )
                guidance_scale = gr.Slider(
                    1, 15,
                    value=float(initial_updates[3]),
                    step=0.5,
                    label="Guidance scale (CFG)",
                    info="How strictly to follow the prompt. Above ~12 tends to overcook colours.",
                )
                with gr.Row(elem_id="seed-row"):
                    with gr.Column(elem_id="seed-field"):
                        seed = gr.Number(
                            value=int(initial_updates[4]),
                            precision=0,
                            label="Seed",
                            info="-1 picks a new random seed each run.",
                        )
                    seed_shuffle = gr.Button(
                        "Shuffle", variant="secondary", elem_id="seed-shuffle",
                    )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    value=initial_updates[5],
                    lines=3,
                    info="Everything the image must not look like. Applies at every step.",
                )
                gr.HTML(
                    f'<p class="ms-note-box">{html.escape(ENHANCEMENT_HELP)}</p>',
                    elem_id="enhancement-help",
                )
                reset_button = gr.Button(
                    "Reset settings", variant="secondary", elem_id="reset-settings",
                )

        # ---------------- chat panel ----------------
        with gr.Column(scale=1, elem_id="chat-panel"):
            with gr.Row(elem_id="chat-header"):
                gr.Markdown(
                    "## Skin image studio\n"
                    "Describe a dermoscopic image. Generations are synthetic "
                    "and not for diagnosis.",
                    elem_id="chat-heading",
                )
                with gr.Row(elem_id="header-controls"):
                    image_type = gr.Radio(
                        choices=list(IMAGE_TYPES),
                        value=initial_updates[0],
                        show_label=False,
                        label="Image type",
                        elem_id="image-type",
                    )
                    medical_prompt_enabled = gr.Checkbox(
                        value=bool(initial_updates[1]),
                        label="Clinical prompt enhancement",
                        elem_id="clinical-toggle",
                    )

            chatbot = gr.Chatbot(
                value=_render_conversation(initial_conversation_id),
                type="messages",
                height=None,
                placeholder=(
                    "**Describe a lesion to begin**\n\n"
                    "Diagnosis, category and body site work best - "
                    "try *melanoma, malignant, back*"
                ),
                show_label=False,
                elem_id="conversation-view",
                layout="bubble",
                bubble_full_width=False,
                sanitize_html=True,
            )

            with gr.Column(elem_id="composer"):
                composer_context = gr.HTML(
                    sync_context("", *initial_updates[:5]), elem_id="composer-context",
                )
                prompt = gr.Textbox(
                    placeholder=(
                        "Diagnosis, category, body site - e.g. "
                        "melanoma, malignant, back"
                    ),
                    label="Message",
                    show_label=False,
                    lines=2,
                    max_lines=6,
                    submit_btn="Generate",
                    elem_id="prompt-box",
                )

    active_conversation = gr.State(initial_conversation_id)
    armed_delete = gr.State(None)

    # Which controls feed the live context strip.
    context_inputs = [
        prompt, image_type, medical_prompt_enabled,
        steps, guidance_scale, seed,
    ]
    settings_inputs = [
        active_conversation, image_type, medical_prompt_enabled,
        steps, guidance_scale, seed, negative_prompt, prompt,
    ]
    # Everything a conversation switch has to refresh. `history_search` is last
    # because only "new conversation" needs to clear it.
    conversation_outputs = [
        chatbot, active_conversation, history_selector, history_empty,
        image_type, medical_prompt_enabled, steps, guidance_scale, seed,
        negative_prompt, composer_context, delete_chat, armed_delete,
        history_search,
    ]
    select_outputs = [
        chatbot, active_conversation, image_type, medical_prompt_enabled,
        steps, guidance_scale, seed, negative_prompt, composer_context,
        delete_chat, armed_delete,
    ]

    prompt.submit(
        generate_image,
        inputs=[
            prompt, active_conversation, image_type, steps, guidance_scale,
            seed, medical_prompt_enabled, negative_prompt,
        ],
        outputs=[
            chatbot, prompt, active_conversation, history_selector,
            history_empty, delete_chat, composer_context,
        ],
        show_api=False,
        show_progress="hidden",  # the in-chat "Generating" bubble replaces it
    )

    new_chat.click(
        start_conversation,
        inputs=[history_search],
        outputs=conversation_outputs,
        show_api=False,
    )

    # `.input` fires only on a real user click. `.change` also fired when the
    # code refreshed the list, which re-rendered the chat mid-generation.
    history_selector.input(
        select_conversation,
        inputs=[history_selector, history_search],
        outputs=select_outputs,
        show_api=False,
    )

    delete_chat.click(
        arm_delete,
        inputs=[active_conversation, armed_delete],
        outputs=conversation_outputs[:-1],
        show_api=False,
    )

    clear_chat.click(
        clear_conversation,
        inputs=[active_conversation, history_search],
        outputs=[chatbot, history_selector, history_empty, armed_delete],
        show_api=False,
    )

    history_search.change(
        filter_history,
        inputs=[history_search, active_conversation],
        outputs=[history_selector, history_empty, active_conversation],
        show_api=False,
    )

    seed_shuffle.click(
        randomize_seed,
        outputs=seed,
        show_api=False,
    ).then(
        persist_settings,
        inputs=settings_inputs,
        outputs=[composer_context],
        show_api=False,
    )

    reset_button.click(
        reset_settings,
        inputs=[active_conversation, prompt],
        outputs=[
            image_type, medical_prompt_enabled, steps, guidance_scale, seed,
            negative_prompt, composer_context,
        ],
        show_api=False,
    )

    # Persist on commit, not on every keystroke or drag frame.
    commit_event = {
        gr.Slider: "release",
        gr.Textbox: "blur",
    }
    for control in (image_type, medical_prompt_enabled, steps, guidance_scale,
                    seed, negative_prompt):
        event = getattr(control, commit_event.get(type(control), "change"))
        event(
            persist_settings,
            inputs=settings_inputs,
            outputs=[composer_context],
            show_api=False,
        )

    prompt.input(
        sync_context,
        inputs=context_inputs,
        outputs=[composer_context],
        show_api=False,
    )

    # The pill starts on "Loading" and settles once warm-up finishes; after
    # that the timer returns gr.skip() so it stops sending anything.
    status_timer = gr.Timer(2.0)
    status_timer.tick(
        model_status_tick,
        outputs=[model_status],
        show_api=False,
    )


def _warm_up():
    """Load the model in the background so the first prompt is not slow."""
    try:
        with _generation_lock:
            load_pipeline()
        _model_state["state"] = "ready"
        _model_state["detail"] = _describe_device()
    except Exception as error:
        print(f"Model warm-up failed: {error}")
        _model_state["state"] = "error"
        _model_state["detail"] = str(error)[:120]


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