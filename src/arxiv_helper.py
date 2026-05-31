"""
arxiv_helper.py  (place inside src/)
Task 4: arXiv Domain Expert Chatbot — Core Logic
--------------------------------------------------
Handles:
  - Loading and filtering the arXiv dataset JSON (CS papers only)
  - Building / loading a FAISS vector database from paper metadata
  - QA chain for answering questions, summarizing and explaining concepts

Dataset setup:
  - Download from https://www.kaggle.com/datasets/Cornell-University/arxiv
  - Place the file arxiv-metadata-oai-snapshot.json inside dataset/arxiv/
"""

import os
import json

from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_HERE            = os.path.dirname(os.path.abspath(__file__))
ARXIV_JSON_PATH  = os.path.join(_HERE, "..", "dataset", "arxiv", "arxiv-metadata-oai-snapshot.json")
ARXIV_VECTORDB   = os.path.join(_HERE, "faiss_arxiv_index")

# ─────────────────────────────────────────────────────────────────────────────
# Shared model instances (same as rest of project)
# ─────────────────────────────────────────────────────────────────────────────
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)

llm = GooglePalm(
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1
)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loader — filters Computer Science papers only
# ─────────────────────────────────────────────────────────────────────────────

def load_arxiv_papers(max_papers: int = 5000) -> list[dict]:
    """
    Read the arXiv JSON file line by line and return CS papers only.

    Each line in the file is one JSON object (paper metadata).
    We filter by categories starting with 'cs.' (Computer Science).

    Parameters
    ----------
    max_papers : how many CS papers to load (default 5000)

    Returns
    -------
    list of dicts with keys: id, title, abstract, categories, authors, year
    """
    if not os.path.exists(ARXIV_JSON_PATH):
        raise FileNotFoundError(
            f"arXiv dataset not found at '{ARXIV_JSON_PATH}'.\n"
            "Please download it from:\n"
            "https://www.kaggle.com/datasets/Cornell-University/arxiv\n"
            "and place the JSON file inside dataset/arxiv/"
        )

    papers = []
    with open(ARXIV_JSON_PATH, "r") as f:
        for line in f:
            if len(papers) >= max_papers:
                break
            try:
                paper = json.loads(line)
                categories = paper.get("categories", "")
                # Keep only Computer Science papers
                if not any(cat.startswith("cs.") for cat in categories.split()):
                    continue
                papers.append({
                    "id":         paper.get("id", ""),
                    "title":      paper.get("title", "").replace("\n", " ").strip(),
                    "abstract":   paper.get("abstract", "").replace("\n", " ").strip(),
                    "categories": categories,
                    "authors":    paper.get("authors", ""),
                    "year":       paper.get("update_date", "")[:4],
                })
            except json.JSONDecodeError:
                continue

    return papers


# ─────────────────────────────────────────────────────────────────────────────
# Vector Database
# ─────────────────────────────────────────────────────────────────────────────

def create_arxiv_vector_db(max_papers: int = 5000) -> int:
    """
    Load CS papers from arXiv JSON, convert to Documents, build and save FAISS index.

    Returns number of papers indexed.
    """
    papers = load_arxiv_papers(max_papers)

    documents = []
    for p in papers:
        content = (
            f"Title: {p['title']}\n"
            f"Categories: {p['categories']}\n"
            f"Year: {p['year']}\n"
            f"Abstract: {p['abstract']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "id":         p["id"],
                "title":      p["title"],
                "categories": p["categories"],
                "year":       p["year"],
                "authors":    p["authors"],
            }
        )
        documents.append(doc)

    vectordb = FAISS.from_documents(documents=documents, embedding=instructor_embeddings)
    vectordb.save_local(ARXIV_VECTORDB)
    return len(documents)


def arxiv_vector_db_exists() -> bool:
    return os.path.isdir(ARXIV_VECTORDB)


# ─────────────────────────────────────────────────────────────────────────────
# QA Chain — Answer questions about CS research
# ─────────────────────────────────────────────────────────────────────────────

_QA_PROMPT = """You are an expert in Computer Science research. 
Use the research paper abstracts below to answer the question clearly and accurately.
If the answer is not in the context, say "I don't have enough information on this topic."
Keep your answer concise and easy to understand.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# ─────────────────────────────────────────────────────────────────────────────
# Summarize Prompt — summarize a specific paper
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_PROMPT = """You are a research paper summarizer.
Based on the paper abstracts below, provide a clear and simple summary.
Cover: what problem it solves, the approach used, and the key findings.
Keep it under 150 words.

CONTEXT:
{context}

PAPER/TOPIC: {question}

SUMMARY:"""

# ─────────────────────────────────────────────────────────────────────────────
# Explain Prompt — explain a concept simply
# ─────────────────────────────────────────────────────────────────────────────

_EXPLAIN_PROMPT = """You are a computer science professor who explains complex topics simply.
Use the research context below to explain the concept in plain English.
Use an analogy if it helps. Keep it under 150 words.

CONTEXT:
{context}

CONCEPT TO EXPLAIN: {question}

EXPLANATION:"""


def _build_chain(prompt_template: str):
    """Build a RetrievalQA chain with the given prompt template."""
    vectordb  = FAISS.load_local(ARXIV_VECTORDB, instructor_embeddings)
    retriever = vectordb.as_retriever(
        score_threshold=0.5,
        search_kwargs={"k": 3}
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def get_arxiv_qa_chain():
    """QA chain — for answering general research questions."""
    return _build_chain(_QA_PROMPT)


def get_summary_chain():
    """Summary chain — for summarizing a paper or topic."""
    return _build_chain(_SUMMARY_PROMPT)


def get_explain_chain():
    """Explain chain — for explaining a concept in simple terms."""
    return _build_chain(_EXPLAIN_PROMPT)


# ─────────────────────────────────────────────────────────────────────────────
# Paper Search — keyword search directly in FAISS
# ─────────────────────────────────────────────────────────────────────────────

def search_papers(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the FAISS index for papers related to the query.
    Returns a list of paper metadata dicts.
    """
    vectordb = FAISS.load_local(ARXIV_VECTORDB, instructor_embeddings)
    results  = vectordb.similarity_search(query, k=top_k)
    papers   = []
    for doc in results:
        papers.append({
            "title":      doc.metadata.get("title", "—"),
            "categories": doc.metadata.get("categories", "—"),
            "year":       doc.metadata.get("year", "—"),
            "authors":    doc.metadata.get("authors", "—"),
            "abstract":   doc.page_content.split("Abstract:")[-1].strip()[:300] + "…",
        })
    return papers
