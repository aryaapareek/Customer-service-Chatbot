# Generative AI Q&A: Customer Service & Medical Chatbot System

An end-to-end LLM project built with **Google Palm**, **Google Gemini**, **LangChain**, and **FAISS**.  
Originally developed as a Q&A system for the e-learning company **Nullclass**, and extended with six advanced AI features as part of a virtual internship.

---

## Project Overview

Nullclass sells data-related courses and virtual internships. Thousands of learners previously had to use Discord or email to get answers from human staff. This system provides a Streamlit-based interface where students can ask questions and get answers instantly — reducing staff workload significantly.

The project has been extended with:
- **Task 1** — Dynamic Knowledge Base Expansion
- **Task 2** — Multi-Modal Chatbot (image understanding + image generation)
- **Task 3** — Medical Q&A Chatbot using the MedQuAD Dataset
- **Task 4** — arXiv Domain Expert Chatbot (Computer Science)
- **Task 5** — Sentiment Analysis Integration
- **Task 6** — Multilingual Support (6 languages)

---

## Project Structure

```
customer_service_bot/
│
├── requirements.txt
├── README.md
├── .env                            ← your Google API key (not committed to git)
│
├── dataset/
│   ├── dataset.csv                 ← Nullclass FAQ dataset (76 QA pairs)
│   ├── medquad/                    ← clone MedQuAD repo here (Task 3)
│   │   ├── 1_CancerGov_QA/
│   │   ├── 2_GARD_QA/
│   │   └── ...
│   └── arxiv/                      ← place arXiv JSON file here (Task 4)
│       └── arxiv-metadata-oai-snapshot.json
│
└── src/
    ├── langchain_helper.py         ← core RAG logic for customer service bot
    ├── main.py                     ← Streamlit UI (core + Tasks 1, 2, 5, 6)
    ├── kb_updater.py               ← Task 1: dynamic knowledge base expansion
    ├── multimodal_helper.py        ← Task 2: image understanding & generation
    ├── medical_ner.py              ← Task 3: medical entity recognition
    ├── medquad_helper.py           ← Task 3: MedQuAD parser, FAISS, QA chain
    ├── medical_app.py              ← Task 3: standalone medical chatbot UI
    ├── arxiv_helper.py             ← Task 4: arXiv parser, FAISS, QA chain
    ├── arxiv_app.py                ← Task 4: standalone arXiv chatbot UI
    ├── sentiment_helper.py         ← Task 5: sentiment detection & tone response
    └── language_helper.py          ← Task 6: language detection & translation
```

> **Auto-generated at runtime** (not committed to git):
> `src/faiss_index/` · `src/faiss_medical_index/` · `src/faiss_arxiv_index/` · `src/pending_updates/` · `src/update_log.json`

---

## Features

### Core — Customer Service Chatbot
- RAG pipeline using **Google Palm** + **LangChain** + **FAISS**
- FAQ knowledge base built from Nullclass's real CSV dataset
- Similarity-score threshold (0.7) ensures only confident answers are returned
- Answers grounded strictly in the FAQ data — no hallucination

### Task 1 — Dynamic Knowledge Base Expansion
- Upload new CSV files through the UI to instantly expand the knowledge base
- Background **auto-update scheduler** (configurable: 1 / 6 / 12 / 24 hours)
- Watches a `pending_updates/` folder and merges new data automatically
- Full update history log with timestamps and document counts
- Processed files archived to `pending_updates/processed/` — never double-processed

### Task 2 — Multi-Modal Chatbot
- **Image Understanding** — upload any image (photo, chart, screenshot) and ask questions about it using **Gemini 1.5 Flash** Vision
- **Image Generation** — describe what you want and get an AI-generated image (free, no extra API key)
- Auto-detects image generation intent from natural language (e.g. *"draw a diagram of…"*)
- Download button for all generated images

### Task 3 — Medical Q&A Chatbot
- Built on the **MedQuAD** dataset — 47,457 QA pairs from 12 NIH websites
- Covers 37 question types: Treatment, Symptoms, Diagnosis, Causes, Prevention, Side Effects, Prognosis, and more
- **Medical Entity Recognition (NER)** — detects symptoms, diseases, treatments, and body parts with colour-coded badges
- Retrieval-augmented answers with NIH source attribution
- Always appends a professional medical consultation reminder
- Runs as a **separate Streamlit app** (`medical_app.py`)

### Task 4 — arXiv Domain Expert Chatbot
- Built on the **arXiv dataset** (Cornell University) — Computer Science papers subset
- Four modes: Q&A, Paper Search, Summarization, and Concept Explanation
- Configurable index size (1,000–50,000 papers)
- Runs as a **separate Streamlit app** (`arxiv_app.py`)

### Task 5 — Sentiment Analysis
- Detects **Positive**, **Negative**, or **Neutral** sentiment in every user message
- Displays a colour-coded sentiment badge with polarity score after each question
- Automatically adjusts the chatbot's tone — empathetic for negative, warm for positive
- Powered by **TextBlob** — lightweight, no GPU required

### Task 6 — Multilingual Support
- Supports **6 languages**: English 🇬🇧, Hindi 🇮🇳, Spanish 🇪🇸, French 🇫🇷, Arabic 🇸🇦, German 🇩🇪
- **Automatic language detection** — no need to manually select
- Translates user question → English → feeds into QA chain → translates answer back
- Culturally appropriate greetings and responses per language
- Manual language selector dropdown also available
- Powered by **langdetect** + **deep-translator** (free, no extra API key)

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/farhangouri1/customer_service_chatbot.git
cd customer_service_chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your Google API key

Get a free API key from [Google AI Studio](https://aistudio.google.com) and create a `.env` file in the project root:

```bash
GOOGLE_API_KEY="your_api_key_here"
```

> The same key works for Google Palm (customer service bot) and Gemini (multimodal + medical).

### 4. (Task 3 only) Clone the MedQuAD dataset

```bash
cd dataset
git clone https://github.com/abachaa/MedQuAD medquad
```

### 5. (Task 4 only) Download the arXiv dataset

- Go to [https://www.kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Download `arxiv-metadata-oai-snapshot.json`
- Place it inside `dataset/arxiv/`

### 6. (Task 5 only) Download TextBlob language data

```bash
python -m textblob.download_corpora
```

---

## Usage

### Customer Service Chatbot (Core + Tasks 1, 2, 5, 6)

```bash
streamlit run src/main.py
```

1. Click **"Create Knowledgebase"** — builds the FAISS index from `dataset.csv`
2. Select a language or leave it on Auto-Detect
3. Type a question in the **Question** box and press Enter
4. View the detected language, sentiment badge, and answer
5. Expand **"Knowledge Base Manager"** to add new data or schedule auto-updates
6. Expand **"Multi-Modal Chat"** to analyse images or generate images

### Medical Q&A Chatbot (Task 3)

```bash
streamlit run src/medical_app.py
```

1. Click **"Build Medical Knowledge Base"** in the sidebar (one-time setup)
2. Type or select a medical question
3. Click **"Get Answer"** — entities are highlighted automatically as you type

### arXiv Expert Chatbot (Task 4)

```bash
streamlit run src/arxiv_app.py
```

1. Click **"Build Knowledge Base"** in the sidebar (one-time setup)
2. Use the four tabs: Ask a Question, Search Papers, Summarize, Explain a Concept

---

## Sample Questions

### Customer Service Bot
- Do you guys provide internship and also do you offer EMI payments?
- Do you have a JavaScript course?
- Should I learn Power BI or Tableau?
- I've a Mac computer. Can I use Power BI on it?
- I don't see Power Pivot. How can I enable it?

### Multilingual Examples (Task 6)
- *"क्या आपके पास JavaScript कोर्स है?"* (Hindi)
- *"¿Ofrecen pasantías?"* (Spanish)
- *"Avez-vous des cours de Python?"* (French)

### Medical Chatbot (Task 3)
- What are the symptoms of diabetes?
- How is hypertension treated?
- What causes Alzheimer's disease?

### arXiv Expert Chatbot (Task 4)
- What is transformer architecture in deep learning?
- How does reinforcement learning work?
- Explain attention mechanism simply

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Palm 2 |
| Vision & Multimodal | Google Gemini 1.5 Flash |
| Embeddings | HuggingFace `hkunlp/instructor-large` |
| Vector Database | FAISS |
| Orchestration | LangChain 0.0.339 |
| UI | Streamlit |
| Medical Dataset | MedQuAD (NIH) |
| Research Dataset | arXiv (Cornell University) |
| Image Generation | Pollinations.ai |
| Sentiment Analysis | TextBlob |
| Language Detection | langdetect |
| Translation | deep-translator (Google Translate) |

---

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API key — used for Palm and Gemini |

---

## Important Notes

- The `.env` file is **not committed** to git. Never share your API key publicly.
- MedQuAD subsets 7, 10, and 11 have answers removed due to MedlinePlus copyright — skipped automatically.
- The medical chatbot is for **educational purposes only** and is not a substitute for professional medical advice.
- Image generation uses [Pollinations.ai](https://pollinations.ai) — free, no additional API key needed.
- Translation uses [deep-translator](https://github.com/nidhaloff/deep-translator) — free, no additional API key needed.
- The arXiv dataset file is large (~3–4 GB). It is not committed to git.

---

## Acknowledgements

- [Nullclass](https://nullclass.com) — for the original FAQ dataset
- [MedQuAD](https://github.com/abachaa/MedQuAD) — Ben Abacha & Demner-Fushman, U.S. National Library of Medicine
- [arXiv](https://arxiv.org) / [Cornell University](https://www.kaggle.com/datasets/Cornell-University/arxiv) — research paper dataset
- [Google AI](https://ai.google.dev) — Palm 2 and Gemini APIs
- [LangChain](https://langchain.com) — LLM orchestration framework
