"""
arxiv_app.py  (place inside src/)
Task 4: arXiv Domain Expert Chatbot — Streamlit UI
----------------------------------------------------
Run with:
    streamlit run src/arxiv_app.py
"""

import streamlit as st
from arxiv_helper import (
    create_arxiv_vector_db,
    arxiv_vector_db_exists,
    get_arxiv_qa_chain,
    get_summary_chain,
    get_explain_chain,
    search_papers,
    ARXIV_JSON_PATH,
)
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="arXiv CS Expert Chatbot",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 arXiv Computer Science Expert")
st.markdown(
    "Ask questions about CS research, search for papers, get summaries, "
    "or understand complex concepts — powered by the **arXiv dataset** and **Google Palm**."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Setup
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # Dataset check
    if os.path.exists(ARXIV_JSON_PATH):
        st.success("✅ arXiv dataset found")
    else:
        st.error("❌ arXiv dataset not found")
        st.markdown(
            "**Download the dataset:**\n"
            "1. Go to [Kaggle arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)\n"
            "2. Download `arxiv-metadata-oai-snapshot.json`\n"
            "3. Place it inside `dataset/arxiv/`"
        )

    # KB check
    if arxiv_vector_db_exists():
        st.success("✅ Knowledge base is ready")
    else:
        st.warning("⚠️ Knowledge base not built yet")

    # Paper count slider
    max_papers = st.slider(
        "Papers to index",
        min_value=1000,
        max_value=50000,
        value=5000,
        step=1000,
        help="More papers = better answers but slower build"
    )

    if st.button("🔨 Build Knowledge Base", use_container_width=True):
        if not os.path.exists(ARXIV_JSON_PATH):
            st.error("Dataset not found. Please download it first.")
        else:
            with st.spinner(f"Loading {max_papers:,} CS papers and building index… this takes a few minutes."):
                try:
                    n = create_arxiv_vector_db(max_papers=max_papers)
                    st.success(f"✅ Done! {n:,} papers indexed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.markdown("---")
    st.markdown(
        "**Dataset:** arXiv (Cornell University)\n\n"
        "**Domain:** Computer Science\n\n"
        "**Covers:** AI, ML, CV, NLP, Systems, Theory, and more"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main Interface — 4 Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Ask a Question",
    "🔍 Search Papers",
    "📄 Summarize",
    "💡 Explain a Concept"
])

# ── Tab 1: Q&A ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("💬 Ask Anything About CS Research")

    SAMPLE_QS = [
        "What is transformer architecture in deep learning?",
        "How does reinforcement learning work?",
        "What are graph neural networks used for?",
        "What is the difference between supervised and unsupervised learning?",
        "What are large language models?",
        "How does attention mechanism work in NLP?",
    ]

    sample = st.selectbox(
        "Try a sample question:",
        ["— type your own —"] + SAMPLE_QS,
        key="qa_sample"
    )
    default = "" if sample.startswith("—") else sample
    question = st.text_input("Your question:", value=default, key="qa_input")

    if st.button("Get Answer", type="primary", key="qa_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not arxiv_vector_db_exists():
            st.error("Please build the knowledge base first using the sidebar.")
        else:
            with st.spinner("Searching research papers…"):
                try:
                    chain    = get_arxiv_qa_chain()
                    response = chain(question)
                    st.markdown("### Answer")
                    st.write(response["result"])

                    # Show source papers
                    if response.get("source_documents"):
                        with st.expander("📚 Source Papers", expanded=False):
                            for doc in response["source_documents"]:
                                st.markdown(
                                    f"**{doc.metadata.get('title', '—')}**  \n"
                                    f"📅 {doc.metadata.get('year', '—')}  |  "
                                    f"🏷️ `{doc.metadata.get('categories', '—')}`"
                                )
                                st.markdown("---")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 2: Paper Search ───────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Search Research Papers")
    st.markdown("Find CS papers related to any topic from the arXiv dataset.")

    search_query = st.text_input(
        "Search topic:",
        placeholder="e.g. object detection, BERT, federated learning",
        key="search_input"
    )
    top_k = st.slider("Number of results", 1, 10, 5, key="search_k")

    if st.button("🔍 Search", key="search_btn"):
        if not search_query.strip():
            st.warning("Please enter a search topic.")
        elif not arxiv_vector_db_exists():
            st.error("Please build the knowledge base first.")
        else:
            with st.spinner("Searching…"):
                try:
                    results = search_papers(search_query, top_k=top_k)
                    if not results:
                        st.info("No papers found for this topic.")
                    else:
                        st.markdown(f"**Found {len(results)} papers:**")
                        for i, p in enumerate(results, 1):
                            with st.expander(f"{i}. {p['title']}", expanded=False):
                                st.markdown(
                                    f"📅 **Year:** {p['year']}  \n"
                                    f"🏷️ **Categories:** `{p['categories']}`  \n"
                                    f"✍️ **Authors:** {p['authors'][:100]}…"
                                )
                                st.markdown("**Abstract:**")
                                st.write(p["abstract"])
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 3: Summarize ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("📄 Summarize a Paper or Topic")
    st.markdown("Enter a paper title or topic and get a short summary.")

    summary_input = st.text_input(
        "Paper title or topic:",
        placeholder="e.g. Attention Is All You Need, GANs, BERT",
        key="summary_input"
    )

    if st.button("📄 Summarize", key="summary_btn"):
        if not summary_input.strip():
            st.warning("Please enter a paper title or topic.")
        elif not arxiv_vector_db_exists():
            st.error("Please build the knowledge base first.")
        else:
            with st.spinner("Generating summary…"):
                try:
                    chain    = get_summary_chain()
                    response = chain(summary_input)
                    st.markdown("### Summary")
                    st.write(response["result"])

                    if response.get("source_documents"):
                        with st.expander("📚 Related Papers Used", expanded=False):
                            for doc in response["source_documents"]:
                                st.markdown(
                                    f"**{doc.metadata.get('title', '—')}**  \n"
                                    f"📅 {doc.metadata.get('year', '—')}"
                                )
                                st.markdown("---")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 4: Explain a Concept ─────────────────────────────────────────────────
with tab4:
    st.subheader("💡 Explain a CS Concept Simply")
    st.markdown("Type any complex CS concept and get a plain English explanation.")

    SAMPLE_CONCEPTS = [
        "Neural networks",
        "Backpropagation",
        "Convolutional neural networks",
        "Transformer architecture",
        "Gradient descent",
        "Overfitting",
        "Attention mechanism",
        "Federated learning",
    ]

    concept_sample = st.selectbox(
        "Try a sample concept:",
        ["— type your own —"] + SAMPLE_CONCEPTS,
        key="explain_sample"
    )
    default_concept = "" if concept_sample.startswith("—") else concept_sample
    concept_input = st.text_input(
        "Concept to explain:",
        value=default_concept,
        key="explain_input"
    )

    if st.button("💡 Explain", key="explain_btn"):
        if not concept_input.strip():
            st.warning("Please enter a concept.")
        elif not arxiv_vector_db_exists():
            st.error("Please build the knowledge base first.")
        else:
            with st.spinner("Generating explanation…"):
                try:
                    chain    = get_explain_chain()
                    response = chain(concept_input)
                    st.markdown("### Explanation")
                    st.write(response["result"])

                    # Concept word display — simple visualization
                    st.markdown("### 🔤 Key Terms")
                    words = [
                        w.strip(".,()[]").lower()
                        for w in concept_input.split()
                        if len(w) > 3
                    ]
                    if words:
                        st.markdown(" &nbsp;".join(
                            f'<span style="background:#dbeafe;color:#1e40af;'
                            f'padding:3px 10px;border-radius:12px;'
                            f'font-size:0.85em;">{w}</span>'
                            for w in words
                        ), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Session History
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🕑 Session History", expanded=False):
    if "arxiv_history" not in st.session_state:
        st.session_state.arxiv_history = []

    if not st.session_state.arxiv_history:
        st.info("No queries yet in this session.")
    else:
        for i, q in enumerate(reversed(st.session_state.arxiv_history), 1):
            st.markdown(f"{i}. {q}")

    if st.button("🗑️ Clear History", key="arxiv_clear"):
        st.session_state.arxiv_history = []
        st.rerun()
