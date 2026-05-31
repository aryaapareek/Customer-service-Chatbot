"""
medical_ner.py  (place inside src/)
Task 3: Medical Entity Recognition
-------------------------------------
Keyword-based NER for medical entities — no extra model download required.
Recognises:
  - Symptoms
  - Diseases / Conditions
  - Treatments / Medications
  - Body Parts / Organs
  - Question Type (what the user is asking about)
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# Medical Entity Dictionaries
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOMS = [
    "pain", "fever", "cough", "fatigue", "nausea", "vomiting", "headache",
    "dizziness", "shortness of breath", "chest pain", "swelling", "rash",
    "diarrhea", "constipation", "bleeding", "inflammation", "itching",
    "weakness", "numbness", "insomnia", "anxiety", "depression", "chills",
    "sweating", "weight loss", "weight gain", "appetite loss", "blurred vision",
    "memory loss", "confusion", "seizure", "tremor", "paralysis", "fainting",
    "palpitations", "wheezing", "sneezing", "runny nose", "sore throat",
    "muscle ache", "joint pain", "back pain", "abdominal pain", "bloating",
    "jaundice", "bruising", "hair loss", "dry mouth", "excessive thirst",
    "frequent urination", "night sweats", "hot flashes", "mood swings",
]

DISEASES = [
    "diabetes", "cancer", "hypertension", "asthma", "arthritis", "alzheimer",
    "parkinson", "stroke", "heart disease", "pneumonia", "tuberculosis",
    "HIV", "AIDS", "hepatitis", "epilepsy", "multiple sclerosis", "lupus",
    "fibromyalgia", "hypothyroidism", "hyperthyroidism", "anemia", "leukemia",
    "lymphoma", "melanoma", "osteoporosis", "scoliosis", "celiac disease",
    "crohn disease", "ulcerative colitis", "irritable bowel", "COPD",
    "emphysema", "bronchitis", "sinusitis", "meningitis", "encephalitis",
    "dementia", "schizophrenia", "bipolar disorder", "ADHD", "autism",
    "psoriasis", "eczema", "acne", "rosacea", "glaucoma", "cataracts",
    "macular degeneration", "kidney disease", "renal failure", "cirrhosis",
    "pancreatitis", "gallstones", "kidney stones", "endometriosis", "PCOS",
    "ovarian cancer", "breast cancer", "prostate cancer", "colon cancer",
    "lung cancer", "thyroid cancer", "skin cancer", "bladder cancer",
    "sepsis", "malaria", "dengue", "typhoid", "cholera", "rabies", "measles",
    "mumps", "chickenpox", "shingles", "herpes", "gonorrhea", "syphilis",
    "chlamydia", "gout", "tendinitis", "bursitis", "carpal tunnel",
    "plantar fasciitis", "herniated disc", "spinal stenosis",
]

TREATMENTS = [
    "surgery", "chemotherapy", "radiation", "medication", "therapy",
    "vaccine", "antibiotic", "antiviral", "immunotherapy", "transplant",
    "dialysis", "physical therapy", "psychotherapy", "insulin", "aspirin",
    "ibuprofen", "steroid", "antihistamine", "antidepressant", "antifungal",
    "blood transfusion", "biopsy", "endoscopy", "colonoscopy", "MRI", "CT scan",
    "X-ray", "ultrasound", "echocardiogram", "angioplasty", "bypass surgery",
    "chemotherapy", "hormone therapy", "targeted therapy", "palliative care",
    "rehabilitation", "occupational therapy", "speech therapy", "laser therapy",
    "cryotherapy", "phototherapy", "acupuncture", "chiropractic", "massage",
    "cognitive behavioral therapy", "CBT", "antipsychotic", "mood stabilizer",
    "anticoagulant", "diuretic", "beta blocker", "ACE inhibitor", "statin",
    "bronchodilator", "inhaler", "nebulizer", "oxygen therapy", "CPAP",
    "pacemaker", "defibrillator", "stent", "prosthetic", "orthotics",
    "vaccination", "immunization", "booster", "infusion", "injection",
    "metformin", "lisinopril", "atorvastatin", "omeprazole", "levothyroxine",
]

BODY_PARTS = [
    "heart", "lung", "liver", "kidney", "brain", "stomach", "bone",
    "muscle", "skin", "blood", "nerve", "spine", "joint", "eye", "ear",
    "throat", "colon", "pancreas", "thyroid", "bladder", "uterus", "ovary",
    "prostate", "testicle", "breast", "lymph node", "artery", "vein",
    "intestine", "esophagus", "gallbladder", "appendix", "tonsil", "spleen",
    "adrenal gland", "pituitary gland", "hypothalamus", "cerebellum",
    "cortex", "retina", "cornea", "eardrum", "nasal cavity", "sinus",
    "trachea", "bronchi", "diaphragm", "rib", "pelvis", "femur", "tibia",
    "cartilage", "tendon", "ligament", "disk", "vertebra", "skull",
    "shoulder", "elbow", "wrist", "knee", "ankle", "hip", "neck",
]

# MedQuAD question types — helps classify what the user is asking
QTYPE_PATTERNS = {
    "treatment":    [r"\btreat\w*\b", r"\btherapy\b", r"\bmedication\b", r"\bcure\b", r"\bmanage\w*\b"],
    "symptoms":     [r"\bsymptom\w*\b", r"\bsign\w*\b", r"\bfeel\w*\b", r"\bindication\w*\b"],
    "diagnosis":    [r"\bdiagnos\w*\b", r"\btest\w*\b", r"\bdetect\w*\b", r"\bscreen\w*\b", r"\bcheck\w*\b"],
    "causes":       [r"\bcause\w*\b", r"\brisk factor\w*\b", r"\bwhy\b", r"\btrigger\w*\b"],
    "prevention":   [r"\bprevent\w*\b", r"\bavoid\w*\b", r"\bprotect\w*\b", r"\breduce risk\b"],
    "prognosis":    [r"\bprogno\w*\b", r"\boutcome\w*\b", r"\bsurvival\b", r"\blife expectancy\b"],
    "side effects": [r"\bside effect\w*\b", r"\badverse\b", r"\bcomplication\w*\b", r"\brisk\w*\b"],
    "information":  [r"\bwhat is\b", r"\bwhat are\b", r"\bdefine\b", r"\bexplain\b", r"\btell me\b"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Core NER Function
# ─────────────────────────────────────────────────────────────────────────────

def recognize_medical_entities(text: str) -> dict:
    """
    Perform keyword-based medical NER on input text.

    Returns
    -------
    dict with keys:
        symptoms    : list of matched symptom terms
        diseases    : list of matched disease/condition terms
        treatments  : list of matched treatment terms
        body_parts  : list of matched body part terms
        question_type: detected question intent (str or None)
    """
    text_lower = text.lower()

    def _find_matches(keyword_list: list) -> list:
        found = []
        for kw in keyword_list:
            # Use word-boundary matching for accuracy
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found.append(kw)
        return found

    # Detect question type
    detected_qtype = None
    for qtype, patterns in QTYPE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                detected_qtype = qtype
                break
        if detected_qtype:
            break

    return {
        "symptoms":      _find_matches(SYMPTOMS),
        "diseases":      _find_matches(DISEASES),
        "treatments":    _find_matches(TREATMENTS),
        "body_parts":    _find_matches(BODY_PARTS),
        "question_type": detected_qtype,
    }


def has_any_entities(entities: dict) -> bool:
    """Return True if at least one medical entity was found."""
    return any(
        bool(v) for k, v in entities.items() if k != "question_type"
    )


def format_entities_html(entities: dict) -> str:
    """
    Build a compact HTML badge display for the Streamlit UI.
    Each entity type gets a different colour.
    """
    COLOR_MAP = {
        "symptoms":   ("#fff3cd", "#856404"),   # yellow
        "diseases":   ("#f8d7da", "#842029"),   # red
        "treatments": ("#d1e7dd", "#0f5132"),   # green
        "body_parts": ("#cfe2ff", "#084298"),   # blue
    }
    LABEL_MAP = {
        "symptoms":   "🤒 Symptom",
        "diseases":   "🏥 Disease/Condition",
        "treatments": "💊 Treatment",
        "body_parts": "🫀 Body Part",
    }

    html_parts = []
    for etype, terms in entities.items():
        if etype == "question_type" or not terms:
            continue
        bg, fg = COLOR_MAP[etype]
        label  = LABEL_MAP[etype]
        badges = " ".join(
            f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:12px;font-size:0.8em;margin:2px;display:inline-block;">'
            f'{t}</span>'
            for t in terms
        )
        html_parts.append(
            f'<div style="margin:4px 0"><strong>{label}:</strong> {badges}</div>'
        )

    if entities.get("question_type"):
        html_parts.append(
            f'<div style="margin:4px 0"><strong>🔍 Query type:</strong> '
            f'<span style="background:#e2d9f3;color:#432874;padding:2px 8px;'
            f'border-radius:12px;font-size:0.8em;">'
            f'{entities["question_type"]}</span></div>'
        )

    return "\n".join(html_parts) if html_parts else ""


def format_entities_text(entities: dict) -> list[str]:
    """Return plain-text lines for each entity group (fallback for non-HTML renders)."""
    LABEL_MAP = {
        "symptoms":      "🤒 Symptoms",
        "diseases":      "🏥 Diseases/Conditions",
        "treatments":    "💊 Treatments",
        "body_parts":    "🫀 Body Parts",
        "question_type": "🔍 Query Type",
    }
    lines = []
    for key, val in entities.items():
        if not val:
            continue
        label = LABEL_MAP.get(key, key)
        if isinstance(val, list):
            lines.append(f"**{label}:** {', '.join(val)}")
        else:
            lines.append(f"**{label}:** {val}")
    return lines
