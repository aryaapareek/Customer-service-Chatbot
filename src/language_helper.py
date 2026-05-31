"""
language_helper.py  (place inside src/)
Task 6: Multilingual Support
------------------------------
Provides:
  - detect_language(text)        : detects language of user input
  - translate_to_english(text)   : translates any language → English
  - translate_from_english(text) : translates English → target language
  - get_language_greeting(code)  : culturally appropriate greeting per language
  - SUPPORTED_LANGUAGES          : dict of supported language codes + names

Supported languages (6 total):
  English (en), Hindi (hi), Spanish (es),
  French (fr), Arabic (ar), German (de)

Uses:
  - langdetect  : for automatic language detection
  - deep-translator : for translation (free, no API key needed)
"""

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# ─────────────────────────────────────────────────────────────────────────────
# Supported Languages
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": {"name": "English",  "flag": "🇬🇧", "native": "English"},
    "hi": {"name": "Hindi",    "flag": "🇮🇳", "native": "हिन्दी"},
    "es": {"name": "Spanish",  "flag": "🇪🇸", "native": "Español"},
    "fr": {"name": "French",   "flag": "🇫🇷", "native": "Français"},
    "ar": {"name": "Arabic",   "flag": "🇸🇦", "native": "العربية"},
    "de": {"name": "German",   "flag": "🇩🇪", "native": "Deutsch"},
}

# Culturally appropriate greetings per language
GREETINGS = {
    "en": "Hello! How can I help you today?",
    "hi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
    "es": "¡Hola! ¿En qué puedo ayudarte hoy?",
    "fr": "Bonjour! Comment puis-je vous aider aujourd'hui?",
    "ar": "مرحباً! كيف يمكنني مساعدتك اليوم؟",
    "de": "Hallo! Wie kann ich Ihnen heute helfen?",
}

# Culturally appropriate sorry/empathy messages per language
EMPATHY_MESSAGES = {
    "en": "I'm sorry to hear that.",
    "hi": "मुझे यह सुनकर खेद है।",
    "es": "Lamento escuchar eso.",
    "fr": "Je suis désolé d'entendre cela.",
    "ar": "أنا آسف لسماع ذلك.",
    "de": "Das tut mir leid zu hören.",
}

# "I don't know" messages per language
DONT_KNOW_MESSAGES = {
    "en": "I don't know the answer to that question.",
    "hi": "मुझे इस प्रश्न का उत्तर नहीं पता।",
    "es": "No sé la respuesta a esa pregunta.",
    "fr": "Je ne connais pas la réponse à cette question.",
    "ar": "لا أعرف إجابة هذا السؤال.",
    "de": "Ich kenne die Antwort auf diese Frage nicht.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(text: str) -> dict:
    """
    Detect the language of the input text.

    Returns
    -------
    dict with keys:
        code       : language code e.g. "hi"
        name       : language name e.g. "Hindi"
        flag       : emoji flag e.g. "🇮🇳"
        native     : native name e.g. "हिन्दी"
        supported  : bool — True if we support this language
    """
    try:
        code = detect(text)
    except LangDetectException:
        code = "en"

    # Normalize — langdetect sometimes returns zh-cn, pt, etc.
    # Map to our supported set or fall back to English
    if code not in SUPPORTED_LANGUAGES:
        code = "en"

    info = SUPPORTED_LANGUAGES[code]
    return {
        "code":      code,
        "name":      info["name"],
        "flag":      info["flag"],
        "native":    info["native"],
        "supported": code in SUPPORTED_LANGUAGES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Translation
# ─────────────────────────────────────────────────────────────────────────────

def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """
    Translate text from any language to English.
    Returns original text if translation fails or source is already English.
    """
    if source_lang == "en":
        return text
    try:
        translated = GoogleTranslator(
            source=source_lang if source_lang != "auto" else "auto",
            target="en"
        ).translate(text)
        return translated or text
    except Exception:
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate English text to the target language.
    Returns original text if target is English or translation fails.
    """
    if target_lang == "en" or not target_lang:
        return text
    try:
        translated = GoogleTranslator(
            source="en",
            target=target_lang
        ).translate(text)
        return translated or text
    except Exception:
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Culturally Appropriate Responses
# ─────────────────────────────────────────────────────────────────────────────

def get_greeting(lang_code: str) -> str:
    """Return a culturally appropriate greeting for the given language."""
    return GREETINGS.get(lang_code, GREETINGS["en"])


def get_empathy_message(lang_code: str) -> str:
    """Return a culturally appropriate empathy message."""
    return EMPATHY_MESSAGES.get(lang_code, EMPATHY_MESSAGES["en"])


def get_dont_know_message(lang_code: str) -> str:
    """Return 'I don't know' in the user's language."""
    return DONT_KNOW_MESSAGES.get(lang_code, DONT_KNOW_MESSAGES["en"])


def localize_dont_know(answer: str, lang_code: str) -> str:
    """
    If the bot says 'I don't know', replace it with
    the localised version in the user's language.
    """
    if "don't know" in answer.lower() or "i don't know" in answer.lower():
        return get_dont_know_message(lang_code)
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline — used by main.py
# ─────────────────────────────────────────────────────────────────────────────

def process_multilingual(question: str, manual_lang: str = "auto") -> dict:
    """
    Full multilingual processing pipeline.

    Steps:
      1. Detect language of the question
      2. Translate question to English
      3. Return translated question + language info for the QA chain

    Parameters
    ----------
    question    : original user question
    manual_lang : language code if user manually selected one, else "auto"

    Returns
    -------
    dict with keys:
        original_question    : str
        english_question     : str  — for feeding into the QA chain
        detected_lang        : dict — language info
    """
    # Detect or use manual selection
    if manual_lang != "auto" and manual_lang in SUPPORTED_LANGUAGES:
        lang_info = {**SUPPORTED_LANGUAGES[manual_lang], "code": manual_lang, "supported": True}
    else:
        lang_info = detect_language(question)

    lang_code        = lang_info["code"]
    english_question = translate_to_english(question, source_lang=lang_code)

    return {
        "original_question": question,
        "english_question":  english_question,
        "detected_lang":     lang_info,
    }


def translate_answer(answer: str, target_lang: str) -> str:
    """
    Translate the English answer back to the user's language.
    Also localises the 'I don't know' phrase if present.
    """
    answer = localize_dont_know(answer, target_lang)
    return translate_from_english(answer, target_lang)
