"""
medical_app.py  (place inside src/)
Task 3: Medical Q&A Chatbot — Streamlit UI
-------------------------------------------
Run with:
    streamlit run src/medical_app.py

This is a SEPARATE app from main.py (the customer-service bot).
It uses the same Google API key from .env.
"""

import streamlit as st
from medquad_helper import (
    create_medical_vector_db,
    get_medical_qa_chain,
    medical_vector_db_exists,
    extract_source_info,
    MEDQUAD_DATA_DIR,
)
from medical_ner import (
    recognize_medical_entities,
    has_any_entities,
    format_entities_html,
    format_entities_text,
)
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🏥",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏥 Medical Q&A Chatbot")
st.markdown(
    "Powered by the **[MedQuAD Dataset](https://github.com/abachaa/MedQuAD)** "
    "(47,457 QA pairs from 12 NIH websites) · Google Palm · LangChain · FAISS"
)

# ─────────────────────────────────────────────────────────────────────────────
# Medical Disclaimer  (always visible)
# ─────────────────────────────────────────────────────────────────────────────
st.warning(
    "⚠️ **Medical Disclaimer:** This chatbot provides general medical information "
    "sourced from NIH for **educational purposes only**. It is **not** a substitute "
    "for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional for any medical concerns."
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Setup & Info
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # Dataset presence check
    dataset_present = os.path.isdir(MEDQUAD_DATA_DIR)
    if dataset_present:
        import glob
        xml_count = len(glob.glob(
            os.path.join(MEDQUAD_DATA_DIR, "**", "*.xml"), recursive=True
        ))
        st.success(f"✅ MedQuAD dataset found\n\n{xml_count} XML files detected")
    else:
        st.error("❌ MedQuAD dataset not found")
        st.markdown(
            "**To set up the dataset:**\n"
            "```bash\n"
            "cd dataset\n"
            "git clone https://github.com/abachaa/MedQuAD medquad\n"
            "```"
        )

    st.markdown("---")

    # Knowledge base status
    if medical_vector_db_exists():
        st.success("✅ Medical knowledge base is ready")
    else:
        st.warning("⚠️ Knowledge base not built yet")

    # Cap slider — helpful on low-RAM machines
    st.markdown("**Index Size** *(lower = faster build)*")
    max_docs = st.slider(
        "Max QA pairs to index",
        min_value=1000,
        max_value=47000,
        value=10000,
        step=1000,
        help="Full dataset = 47,457 pairs. Reduce if you run into memory issues."
    )

    if st.button("🔨 Build Medical Knowledge Base", use_container_width=True):
        if not dataset_present:
            st.error(
                "MedQuAD dataset not found. "
                "Clone the repo into dataset/medquad/ first."
            )
        else:
            with st.spinner(
                f"Parsing XML files and indexing up to {max_docs:,} QA pairs… "
                "This may take a few minutes."
            ):
                try:
                    n = create_medical_vector_db(max_docs=max_docs)
                    st.success(f"✅ Knowledge base built — {n:,} documents indexed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Build failed: {e}")

    st.markdown("---")
    st.markdown(
        "**MedQuAD Data Sources**\n"
        "- NCI / Cancer.gov\n"
        "- GARD (Rare Diseases)\n"
        "- MedlinePlus Health Topics\n"
        "- NIDDK · NINDS · NHLBI\n"
        "- CDC · NIH Senior Health\n"
        "- MedlinePlus Drugs & Herbs"
    )
    st.markdown(
        "**Covers 37 Question Types**\n"
        "Treatment · Symptoms · Diagnosis · Causes\n"
        "Prevention · Side Effects · Prognosis …"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main Q&A Interface
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
    "What causes Alzheimer's disease?",
    "How can I prevent heart disease?",
    "What are the side effects of chemotherapy?",
    "What is multiple sclerosis?",
    "How is asthma diagnosed?",
    "What are the treatments for arthritis?",
    "What are the risk factors for stroke?",
]

st.subheader("💬 Ask a Medical Question")

# Sample question picker
selected_sample = st.selectbox(
    "Try a sample question:",
    options=["— type your own question below —"] + SAMPLE_QUESTIONS,
    key="sample_selector"
)

# Main text input — pre-fills if sample is selected
default_q = "" if selected_sample.startswith("—") else selected_sample
medical_question = st.text_input(
    "Your question:",
    value=default_q,
    placeholder="e.g. What are the treatment options for Type 2 diabetes?",
    key="med_question"
)

# ─────────────────────────────────────────────────────────────────────────────
# Entity Recognition — runs as user types (no button needed)
# ─────────────────────────────────────────────────────────────────────────────
if medical_question.strip():
    entities = recognize_medical_entities(medical_question)

    if has_any_entities(entities):
        with st.expander("🔬 Recognised Medical Entities", expanded=True):
            entity_html = format_entities_html(entities)
            if entity_html:
                st.markdown(entity_html, unsafe_allow_html=True)
            else:
                for line in format_entities_text(entities):
                    st.markdown(line)

# ─────────────────────────────────────────────────────────────────────────────
# Answer Retrieval
# ─────────────────────────────────────────────────────────────────────────────
ask_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)

if ask_btn:
    if not medical_question.strip():
        st.warning("Please enter a question first.")

    elif not medical_vector_db_exists():
        st.error(
            "Medical knowledge base not found. "
            "Please build it first using the sidebar button."
        )

    else:
        with st.spinner("Searching MedQuAD knowledge base…"):
            try:
                chain    = get_medical_qa_chain()
                response = chain(medical_question)

                answer       = response.get("result", "")
                source_docs  = response.get("source_documents", [])

                # ── Answer ───────────────────────────────────────────────────
                st.markdown("---")
                st.subheader("📋 Answer")
                st.write(answer)

                # ── Source Documents ─────────────────────────────────────────
                if source_docs:
                    with st.expander("📚 Source Documents from MedQuAD", expanded=False):
                        sources = extract_source_info(source_docs)
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}. {src['focus']}**  \n"
                                f"🏷️ Type: `{src['qtype']}`  &nbsp;|&nbsp;  "
                                f"📖 Source: `{src['source']}`"
                            )
                            if src["snippet"]:
                                st.caption(src["snippet"])
                            st.markdown("---")

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"An error occurred: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Chat History  (session-based, resets on refresh)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🕑 Session History", expanded=False):
    if "med_history" not in st.session_state:
        st.session_state.med_history = []

    # Save to history when answer is generated
    if ask_btn and medical_question.strip() and medical_vector_db_exists():
        st.session_state.med_history.append(medical_question)

    if not st.session_state.med_history:
        st.info("No questions asked yet in this session.")
    else:
        for i, q in enumerate(reversed(st.session_state.med_history), 1):
            st.markdown(f"{i}. {q}")

    if st.button("🗑️ Clear History"):
        st.session_state.med_history = []
        st.rerun()
