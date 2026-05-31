"""
multimodal_helper.py  (place inside src/)
Task 2: Multi-Modal Chatbot
----------------------------
Provides:
  - analyze_image(image_bytes, question)  : Gemini Vision → text answer about an image
  - generate_image(prompt)                : Pollinations.ai → image bytes (free, no extra key)
  - should_generate_image(text)           : detects if the user wants an image generated
  - extract_image_prompt(text)            : pulls the subject out of the user's request

Same GOOGLE_API_KEY from .env is reused — no new key needed for Gemini.
"""

import os
import io
import requests
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── Configure Gemini with the same key already in .env ───────────────────────
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ── Gemini model for vision + text ───────────────────────────────────────────
# gemini-1.5-flash supports both images and text (gemini-pro-vision is deprecated)
VISION_MODEL = "gemini-1.5-flash"

# ── Keywords that signal the user wants an image generated ───────────────────
IMAGE_GEN_TRIGGERS = [
    "generate image",
    "generate a image",
    "create image",
    "create a image",
    "make image",
    "make a image",
    "draw",
    "show image",
    "generate picture",
    "create picture",
    "make picture",
    "visualize",
    "produce image",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Image Understanding  →  Gemini 1.5 Flash (Vision)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image(image_bytes: bytes, question: str = "") -> str:
    """
    Send an uploaded image (+ optional question) to Gemini Vision.
    Returns a detailed text response.

    Parameters
    ----------
    image_bytes : raw bytes of the uploaded image
    question    : user's question about the image; defaults to a general description

    Returns
    -------
    str  — Gemini's text answer
    """
    try:
        model = genai.GenerativeModel(VISION_MODEL)
        pil_image = Image.open(io.BytesIO(image_bytes))

        prompt = question.strip() if question.strip() else (
            "Please describe this image in detail. "
            "Mention any text, objects, charts, or concepts you see."
        )

        response = model.generate_content([prompt, pil_image])
        return response.text

    except Exception as exc:
        return f"⚠️ Image analysis failed: {str(exc)}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Image Generation  →  Pollinations.ai  (free · no extra API key)
# ─────────────────────────────────────────────────────────────────────────────

def generate_image(prompt: str, width: int = 512, height: int = 512) -> bytes | None:
    """
    Generate an image from a text prompt using Pollinations.ai.
    Returns raw image bytes on success, or None on failure.

    Parameters
    ----------
    prompt : description of the image to generate
    width  : output image width  (default 512)
    height : output image height (default 512)
    """
    try:
        encoded = requests.utils.quote(prompt)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&nologo=true&seed=42"
        )
        resp = requests.get(url, timeout=45)
        if resp.status_code == 200:
            return resp.content
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Intent Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def should_generate_image(text: str) -> bool:
    """Return True if the user's message is requesting image generation."""
    lower = text.lower()
    return any(trigger in lower for trigger in IMAGE_GEN_TRIGGERS)


def extract_image_prompt(text: str) -> str:
    """
    Pull the subject for image generation out of the user's message.
    e.g. "generate image of a sunset" → "a sunset"
    Falls back to the full text if no trigger phrase is found.
    """
    lower = text.lower()
    for trigger in IMAGE_GEN_TRIGGERS:
        if trigger in lower:
            idx = lower.find(trigger) + len(trigger)
            remainder = text[idx:].strip().lstrip("of for a an :- ").strip()
            return remainder if remainder else text
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 4. Combined multimodal entry-point (used by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def multimodal_chat(text_input: str, image_bytes: bytes | None = None) -> dict:
    """
    Route the user's input to the right capability and return a unified result.

    Returns a dict with keys:
      - mode         : "image_analysis" | "image_generation" | "text"
      - text_answer  : str (always present)
      - image_bytes  : bytes | None  (present for image_generation)
    """
    # ── Case 1: User uploaded an image → analyse it ──────────────────────────
    if image_bytes is not None:
        answer = analyze_image(image_bytes, text_input)
        return {
            "mode": "image_analysis",
            "text_answer": answer,
            "image_bytes": None,
        }

    # ── Case 2: User wants an image generated ────────────────────────────────
    if should_generate_image(text_input):
        img_prompt = extract_image_prompt(text_input)
        img_bytes  = generate_image(img_prompt)
        caption    = (
            f"Here is the generated image for: **{img_prompt}**"
            if img_bytes
            else "⚠️ Image generation failed. Please try again."
        )
        return {
            "mode": "image_generation",
            "text_answer": caption,
            "image_bytes": img_bytes,
        }

    # ── Case 3: Plain text → caller should fall through to existing QA chain ─
    return {
        "mode": "text",
        "text_answer": "",
        "image_bytes": None,
    }
