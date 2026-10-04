"""Turn a short user description into a caption the fine-tuned model understands.

Why this module exists
----------------------
The model is a Stable Diffusion 1.5 fine-tune, and CLIP's text encoder only has
positional embeddings for 77 tokens. Anything longer is either truncated in
silence or blows up with "Token indices sequence length is longer than the
specified maximum sequence length".

Worse, the fine-tune in `train.py` learned captions shaped like::

    dermoscopy image of melanoma, malignant, back
    smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 3

That is roughly 10-15 tokens. Piling a page of clinical instructions on top of a
fine-tune does not make it more medical, it just pushes the prompt out of
distribution *and* out of the token budget.

So "clinical prompt enhancement" here means: parse what the clinician actually
said, normalise it into the caption grammar the model was trained on, and add a
short imaging-modality tail -- all inside a hard token budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# CLIP hard-codes 77 positions. We stay under it on purpose so that the
# BOS/EOS bookkeeping and diffusers' internal max_length can never collide.
CLIP_TOKEN_LIMIT = 75

# The caption prefix used for each capture view. These strings are copied
# verbatim from `train.py` -> `DatasetSpec.caption_prefix`.
IMAGE_TYPES: dict[str, str] = {
    "Dermoscopy": "dermoscopy image of",
    "Smartphone": "smartphone photo of",
    "Clinical photo": "clinical photo of",
}

# A short modality tail per view. This is what actually buys image quality --
# the modality keywords the fine-tune saw, not a wall of prose. Ordered by
# usefulness: if the budget is tight the tail is what gets dropped.
MODALITY_TAILS: dict[str, tuple[str, ...]] = {
    "Dermoscopy": (
        "clinical dermoscopy",
        "polarized illumination",
        "10x magnification",
        "sharp focus",
        "pigment network and skin texture visible",
    ),
    "Smartphone": (
        "handheld smartphone photograph",
        "natural daylight",
        "close-up",
        "natural depth of field",
        "realistic skin texture",
    ),
    "Clinical photo": (
        "clinical photograph",
        "even lighting",
        "realistic framing and distance",
        "realistic skin texture",
    ),
}

# Diagnosis lexicon grounded in `train.py` -> `DATASETS[*].label_map` plus the
# Fitzpatrick17k `label` column. (canonical name, category, aliases)
DIAGNOSES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "actinic keratosis or intraepithelial carcinoma",
        "pre-malignant",
        ("actinic keratosis or intraepithelial carcinoma", "actinic keratosis",
         "intraepithelial carcinoma", "akiec", "ack", "solar keratosis"),
    ),
    ("basal cell carcinoma", "malignant",
     ("basal cell carcinoma", "basal cell", "bcc")),
    ("squamous cell carcinoma", "malignant",
     ("squamous cell carcinoma", "squamous cell", "scc")),
    ("Bowen's disease", "malignant",
     ("bowen's disease", "bowens disease", "bowen's", "bowen")),
    ("melanoma", "malignant",
     ("malignant melanoma", "melanoma", "mel")),
    ("melanocytic nevus", "benign",
     ("melanocytic nevus", "melanocytic naevus", "nevus", "naevus", "mole", "nev")),
    ("seborrheic keratosis", "benign",
     ("seborrheic keratosis", "seborrhoeic keratosis", "seborrheic keratosis-like lesion", "sek")),
    ("benign keratosis-like lesion", "benign",
     ("benign keratosis-like lesion", "benign keratosis like lesion", "solar lentigo",
      "lentigo", "bkl")),
    ("dermatofibroma", "benign", ("dermatofibroma", "df")),
    ("vascular lesion", "benign",
     ("vascular lesion", "angioma", "haemangioma", "hemangioma",
      "pyogenic granuloma", "vasc")),
)

# Categories as spelled in the dataset metadata. Longest first so that
# "pre-malignant" is not shadowed by "malignant".
CATEGORY_TERMS: dict[str, str] = {
    "non-neoplastic": "non-neoplastic",
    "non neoplastic": "non-neoplastic",
    "pre-malignant": "pre-malignant",
    "pre malignant": "pre-malignant",
    "premalignant": "pre-malignant",
    "precancerous": "pre-malignant",
    "malignant": "malignant",
    "benign": "benign",
}

# Body sites: HAM10000 `localization` + the common PAD-UFES-20 `region` values,
# normalised to lowercase and mapped to a canonical phrase.
BODY_SITES: dict[str, str] = {
    "scalp": "scalp", "face": "face", "neck": "neck", "trunk": "torso",
    "torso": "torso", "chest": "chest", "back": "back", "abdomen": "abdomen",
    "upper extremity": "upper extremity", "lower extremity": "lower extremity",
    "arm": "arm", "left arm": "arm", "right arm": "arm",
    "arms": "arm", "forearm": "forearm", "hand": "hand", "hands": "hand",
    "finger": "finger", "fingers": "finger", "wrist": "wrist",
    "leg": "leg", "left leg": "leg", "right leg": "leg",
    "legs": "leg", "lower leg": "lower leg", "thigh": "thigh",
    "knee": "knee", "knees": "knee", "foot": "foot", "feet": "foot",
    "toe": "toe", "toes": "toe", "ankle": "ankle", "elbow": "elbow",
    "shoulder": "shoulder", "buttock": "buttock", "lip": "lip",
    "lips": "lip", "ear": "ear", "ears": "ear", "eye": "eye", "eyes": "eye",
    "beard": "beard", "mouth": "mouth", "mucosa": "mucosa", "genital": "genital",
    "nail": "nail", "nails": "nail",
}

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}

# Modifier words that carry no visual signal and only eat into the budget.
_STOPWORDS = frozenset("""
a an the of on in with and or for to at by from is are be please generate image
images picture pictures photo photos photograph photographs rendering render show
me create make want need would like some real realistic looks look what this that
these those it its as into about shot shots output result results
""".split())


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

_TOKENIZER_DIR: Path | None = None


def set_tokenizer_dir(path) -> None:
    """Point the token counter at the CLIP tokenizer shipped with the model."""
    global _TOKENIZER_DIR
    _TOKENIZER_DIR = Path(path) if path else None
    # A new directory means a possibly different vocabulary.
    _tokenizer.cache_clear()


@lru_cache(maxsize=1)
def _tokenizer():
    """Load the CLIP tokenizer lazily; returns None when it is unavailable."""
    if _TOKENIZER_DIR is None:
        return None
    try:
        from transformers import CLIPTokenizer

        return CLIPTokenizer.from_pretrained(
            str(_TOKENIZER_DIR), local_files_only=True
        )
    except Exception:
        return None


def _estimate_tokens(text: str) -> int:
    """Cheap fallback for when the tokenizer cannot be loaded.

    CLIP's BPE averages a little over three characters per token on English
    clinical words, and never exceeds the word count for short captions.
    """
    return max(1, round(len(text) / 3.6))


def count_tokens(text: str) -> int:
    """Number of CLIP tokens `text` occupies. Always safe to call."""
    text = (text or "").strip()
    if not text:
        return 0
    tokenizer = _tokenizer()
    if tokenizer is None:
        return _estimate_tokens(text)
    try:
        return len(tokenizer(text)["input_ids"])
    except Exception:
        return _estimate_tokens(text)


def truncate_to_limit(
    text: str,
    limit: int = CLIP_TOKEN_LIMIT,
) -> tuple[str, bool]:
    """Shorten `text` to at most `limit` CLIP tokens.

    Whole words are dropped from the end, so the caption prefix -- the part the
    fine-tune is most sensitive to -- always survives. Returns the new text and
    whether anything was removed.
    """
    text = (text or "").strip()
    if count_tokens(text) <= limit:
        return text, False

    words = text.split()
    while words:
        words.pop()
        if not words:
            break
        candidate = " ".join(words).rstrip(" ,;:-")
        if count_tokens(candidate) <= limit:
            return candidate, True

    return "", True


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("_", " ").replace("/", " ")
    return re.sub(r"\s+", " ", text)


def _find_phrase(text: str, phrase: str) -> tuple[int, int] | None:
    """Locate `phrase` in `text` on word boundaries."""
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    match = re.search(pattern, text)
    return (match.start(), match.end()) if match else None


def _consume(text: str, phrases: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Remove every matching phrase from `text`, returning what was matched.

    Longer phrases are tried first so that "upper extremity" wins over
    "extremity" and "left arm" over "arm".
    """
    found: dict[str, str] = {}
    for phrase in sorted(phrases, key=len, reverse=True):
        span = _find_phrase(text, phrase)
        if span:
            start, end = span
            # Keyed by the canonical value so the longest synonym wins and the
            # caption always uses the phrasing the fine-tune was trained on.
            found.setdefault(phrases[phrase], phrase)
            text = text[:start] + " " + text[end:]
    return text, found


def parse_description(text: str) -> dict[str, str]:
    """Split a free-text description into its caption components.

    Anything the lexicon does not recognise is preserved under ``extra`` so a
    user asking for "irregular borders on the shoulder" still gets both the
    shoulder normalisation and their own words.
    """
    original = _normalise(text)

    # A caption already in training format is respected as-is.
    for prefix in IMAGE_TYPES.values():
        span = _find_phrase(original, prefix)
        if span:
            original = original[span[1]:]
            break

    remainder, sites = _consume(original, BODY_SITES)

    remainder, categories = _consume(remainder, {
        key: value for key, value in CATEGORY_TERMS.items()
    })

    diagnosis = ""
    diagnosis_category = ""
    for name, category, aliases in DIAGNOSES:
        found = [a for a in aliases if _find_phrase(remainder, a)]
        if found:
            # Prefer the longest alias: "malignant melanoma" over "mel".
            found.sort(key=len, reverse=True)
            best = found[0]
            for alias in sorted(aliases, key=len, reverse=True):
                span = _find_phrase(remainder, alias)
                if span:
                    remainder = remainder[:span[0]] + " " + remainder[span[1]:]
                    break
            diagnosis = name
            diagnosis_category = category
            break

    remainder, skin_spelled = _consume(remainder, {
        f"fitzpatrick skin type {n}": f"Fitzpatrick skin type {n}"
        for n in _ROMAN.values()
    })
    remainder, skin_roman = _consume(remainder, {
        f"skin type {roman}": f"Fitzpatrick skin type {_ROMAN[roman]}"
        for roman in _ROMAN
    })
    remainder, skin_numeric = _consume(remainder, {
        f"fitzpatrick {n}": f"Fitzpatrick skin type {n}"
        for n in _ROMAN.values()
    })
    skin = next(
        iter(found.values()),
        "",
    ) if (found := skin_spelled or skin_roman or skin_numeric) else ""

    extra_words = [
        word
        for word in re.findall(r"[a-z0-9']+", remainder)
        if word not in _STOPWORDS and len(word) > 1
    ]

    category = next(iter(categories.values()), "")
    if not category:
        category = diagnosis_category

    return {
        "diagnosis": diagnosis,
        "category": category,
        "site": next(iter(sites), ""),
        "skin": skin,
        "extra": " ".join(dict.fromkeys(extra_words)),
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class PromptPlan:
    """Everything the UI needs to explain what will be sent to the model."""

    text: str
    tokens: int
    limit: int = CLIP_TOKEN_LIMIT
    enhanced: bool = False
    truncated: bool = False
    diagnosis: str = ""
    category: str = ""
    site: str = ""
    skin: str = ""
    extra: str = ""
    tail: str = ""

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.tokens)

    @property
    def usage(self) -> float:
        return min(1.0, self.tokens / self.limit) if self.limit else 0.0

    @property
    def preview(self) -> str:
        """One-line summary of how the caption was assembled."""
        if not self.enhanced:
            return "Sent verbatim -- clinical enhancement is off."
        bits = [
            f"diagnosis **{self.diagnosis}**" if self.diagnosis else "no diagnosis match",
            f"category **{self.category}**" if self.category else "no category match",
            f"site **{self.site}**" if self.site else "no site match",
            f"skin **{self.skin}**" if self.skin else "no skin type",
            "modality tail applied" if self.tail else "tail dropped (budget)",
        ]
        return " - ".join(bits)

    def as_caption(self) -> str:
        """Metadata line stored alongside the image in the transcript."""
        parts = []
        if self.diagnosis:
            parts.append(self.diagnosis)
        if self.category:
            parts.append(self.category)
        if self.site:
            parts.append(self.site)
        return " - ".join(parts) if parts else "custom prompt"


def build_plan(
    prompt: str,
    enhanced: bool = True,
    image_type: str = "Dermoscopy",
    limit: int = CLIP_TOKEN_LIMIT,
) -> PromptPlan:
    """Build a token-safe prompt.

    With ``enhanced`` the description is normalised into the training caption
    grammar and a modality tail is appended if the budget allows. Without it the
    user's text is passed through untouched. Either way the result is guaranteed
    to fit ``limit`` CLIP tokens, which is what keeps the tokenizer warning away.
    """
    prompt = (prompt or "").strip()
    prefix = IMAGE_TYPES.get(image_type, IMAGE_TYPES["Dermoscopy"])

    if not prompt:
        return PromptPlan(text="", tokens=0, limit=limit, enhanced=enhanced)

    if not enhanced:
        text, truncated = truncate_to_limit(prompt, limit)
        return PromptPlan(
            text=text,
            tokens=count_tokens(text),
            limit=limit,
            enhanced=False,
            truncated=truncated,
        )

    fields = parse_description(prompt)

    # Build the caption in the exact shape train.py produced.
    subject = fields["diagnosis"] or fields["extra"] or "skin lesion"
    core = f"{prefix} {subject}"
    if fields["category"] and fields["category"] not in core:
        core += f", {fields['category']}"
    if fields["site"] and fields["site"] not in core:
        core += f", {fields['site']}"
    if fields["skin"] and fields["skin"] not in core:
        core += f", {fields['skin']}"
    if fields["extra"] and fields["diagnosis"] and fields["extra"] not in core:
        core += f", {fields['extra']}"

    core, _ = truncate_to_limit(core, limit)

    text = core
    tail_parts: list[str] = []
    for candidate in MODALITY_TAILS.get(image_type, ()):
        attempt = ", ".join([core, *tail_parts, candidate])
        if count_tokens(attempt) <= limit:
            tail_parts.append(candidate)
            text = attempt

    truncated = False
    if count_tokens(text) > limit:  # defensive: core alone must fit
        text, truncated = truncate_to_limit(text, limit)

    return PromptPlan(
        text=text,
        tokens=count_tokens(text),
        limit=limit,
        enhanced=True,
        truncated=truncated,
        tail=", ".join(tail_parts),
        **fields,
    )