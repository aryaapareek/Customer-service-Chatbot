"""
sentiment_helper.py  (place inside src/)
Task 5: Sentiment Analysis Integration
----------------------------------------
Detects customer emotion in every message and returns:
  - sentiment label  : Positive / Negative / Neutral
  - polarity score   : -1.0 (very negative) to +1.0 (very positive)
  - emoji            : visual indicator
  - tone prefix      : prepended to the chatbot answer to feel empathetic
  - color            : for UI badge styling

Uses TextBlob (lightweight, no GPU needed) — just add textblob to requirements.txt.
"""

from textblob import TextBlob

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Thresholds
# ─────────────────────────────────────────────────────────────────────────────
POSITIVE_THRESHOLD =  0.15
NEGATIVE_THRESHOLD = -0.15

# ─────────────────────────────────────────────────────────────────────────────
# Tone Prefixes — prepended to the bot answer based on detected emotion
# ─────────────────────────────────────────────────────────────────────────────
TONE_PREFIXES = {
    "Positive": [
        "Great to hear you're feeling good! 😊 ",
        "Glad you're having a positive experience! ",
        "Wonderful! Here's what I found for you: ",
    ],
    "Negative": [
        "I'm sorry to hear you're having trouble. Let me help you with that. ",
        "I understand your frustration and I'm here to help. ",
        "I'm sorry about that! Here's what I can do for you: ",
    ],
    "Neutral": [
        "",   # neutral gets no prefix — just answer naturally
        "",
        "",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Core Sentiment Detection
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    """
    Analyse the sentiment of a user message using TextBlob.

    Returns
    -------
    dict with keys:
        label      : "Positive" | "Negative" | "Neutral"
        polarity   : float between -1.0 and 1.0
        subjectivity: float between 0.0 (objective) and 1.0 (subjective)
        emoji      : str  — visual indicator
        color      : str  — hex color for UI badge background
        text_color : str  — hex color for badge text
        prefix     : str  — empathetic tone prefix for the bot response
    """
    blob       = TextBlob(str(text))
    polarity   = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    # Classify
    if polarity >= POSITIVE_THRESHOLD:
        label      = "Positive"
        emoji      = "😊"
        color      = "#d1fae5"   # light green
        text_color = "#065f46"
    elif polarity <= NEGATIVE_THRESHOLD:
        label      = "Negative"
        emoji      = "😟"
        color      = "#fee2e2"   # light red
        text_color = "#991b1b"
    else:
        label      = "Neutral"
        emoji      = "😐"
        color      = "#f3f4f6"   # light grey
        text_color = "#374151"

    # Pick a prefix (rotate based on polarity magnitude for variety)
    options = TONE_PREFIXES[label]
    idx     = min(int(abs(polarity) * 10) % len(options), len(options) - 1)
    prefix  = options[idx]

    return {
        "label":        label,
        "polarity":     polarity,
        "subjectivity": subjectivity,
        "emoji":        emoji,
        "color":        color,
        "text_color":   text_color,
        "prefix":       prefix,
    }


def apply_sentiment_to_answer(answer: str, sentiment: dict) -> str:
    """
    Prepend the empathetic tone prefix to the chatbot's answer.
    Only adds prefix for Positive and Negative — Neutral stays as-is.
    """
    prefix = sentiment.get("prefix", "")
    if prefix:
        return prefix + answer
    return answer


def sentiment_badge_html(sentiment: dict) -> str:
    """
    Returns an HTML string for a styled sentiment badge to display in Streamlit.
    """
    label  = sentiment["label"]
    emoji  = sentiment["emoji"]
    score  = sentiment["polarity"]
    bg     = sentiment["color"]
    fg     = sentiment["text_color"]
    return (
        f'<span style="background:{bg};color:{fg};padding:4px 12px;'
        f'border-radius:16px;font-size:0.85em;font-weight:600;">'
        f'{emoji} {label} (score: {score:+.2f})</span>'
    )


def get_sentiment_feedback(sentiment: dict) -> str:
    """
    Returns a short human-readable description of the sentiment result
    for display below the badge.
    """
    label = sentiment["label"]
    score = sentiment["polarity"]
    subj  = sentiment["subjectivity"]

    subj_text = (
        "very subjective/emotional"  if subj > 0.7
        else "somewhat subjective"   if subj > 0.4
        else "fairly objective"
    )

    if label == "Positive":
        return f"Your message has a positive tone ({subj_text}). Score: {score:+.2f}"
    elif label == "Negative":
        return f"Your message seems frustrated or concerned ({subj_text}). Score: {score:+.2f}"
    else:
        return f"Your message has a neutral tone ({subj_text}). Score: {score:+.2f}"
