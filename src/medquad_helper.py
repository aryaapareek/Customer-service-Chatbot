"""
medquad_helper.py  (place inside src/)
Task 3: Medical Q&A — Core Logic
----------------------------------
Handles:
  - Parsing MedQuAD XML files (all 12 NIH subfolders)
  - Building / loading a FAISS vector database from the parsed QA pairs
  - Returning a RetrievalQA chain that answers medical questions

MedQuAD folder structure expected at:
    customer_service_bot/
    └── dataset/
        └── medquad/          ← clone https://github.com/abachaa/MedQuAD here
            ├── 1_CancerGov_QA/
            ├── 2_GARD_QA/
            ├── ...
            └── 12_MPlusHerbsSupplements_QA/

Note: Subsets 7, 10, 11 have no answers (MedlinePlus copyright) — they are
      skipped automatically during parsing.
"""

import os
import glob
import xml.etree.ElementTree as ET

from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths  (resolved relative to this file so they work regardless of cwd)
# ─────────────────────────────────────────────────────────────────────────────
_HERE               = os.path.dirname(os.path.abspath(__file__))
MEDQUAD_DATA_DIR    = os.path.join(_HERE, "..", "dataset", "medquad")
MEDICAL_VECTORDB    = os.path.join(_HERE, "faiss_medical_index")

# ─────────────────────────────────────────────────────────────────────────────
# Shared model instances
# ─────────────────────────────────────────────────────────────────────────────
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)

llm = GooglePalm(
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1
)

# ─────────────────────────────────────────────────────────────────────────────
# MedQuAD XML Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_xml_file(xml_path: str) -> list[dict]:
    """
    Parse one MedQuAD XML file.

    Expected structure:
        <Document source="..." url="...">
          <Focus>Disease or Drug name</Focus>
          <FocusAnnotations>
            <UMLS>
              <SemanticTypes><SemanticType>...</SemanticType></SemanticTypes>
            </UMLS>
          </FocusAnnotations>
          <QAPairs>
            <QAPair pid="1">
              <Question qid="..." qtype="treatment">...</Question>
              <Answer>...</Answer>
            </QAPair>
          </QAPairs>
        </Document>

    Returns list of dicts, one per valid QA pair.
    """
    results = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        focus  = (root.findtext("Focus") or "").strip()
        source = root.get("source", os.path.basename(os.path.dirname(xml_path)))

        # UMLS semantic type (optional metadata)
        sem_type_el = root.find(".//SemanticType")
        semantic_type = sem_type_el.text.strip() if sem_type_el is not None else ""

        for qa_pair in root.findall(".//QAPair"):
            q_el = qa_pair.find("Question")
            a_el = qa_pair.find("Answer")

            if q_el is None or a_el is None:
                continue

            question = (q_el.text or "").strip()
            answer   = (a_el.text or "").strip()
            qtype    = q_el.get("qtype", "").strip()

            # Skip pairs with empty answers (subsets 7, 10, 11)
            if not question or not answer:
                continue

            results.append({
                "question":      question,
                "answer":        answer,
                "focus":         focus,
                "source":        source,
                "qtype":         qtype,
                "semantic_type": semantic_type,
            })

    except ET.ParseError:
        pass  # skip malformed XML files silently
    except Exception:
        pass

    return results


def load_medquad_data(data_dir: str = MEDQUAD_DATA_DIR) -> list[dict]:
    """
    Walk the MedQuAD directory tree and parse every XML file found.

    Returns
    -------
    list of dicts  — all valid QA pairs across all 12 NIH subfolders
    """
    xml_files = glob.glob(os.path.join(data_dir, "**", "*.xml"), recursive=True)

    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in '{data_dir}'.\n"
            "Please clone the MedQuAD repository into dataset/medquad/:\n"
            "  cd dataset && git clone https://github.com/abachaa/MedQuAD medquad"
        )

    all_qa = []
    for xml_file in xml_files:
        all_qa.extend(_parse_xml_file(xml_file))

    return all_qa


# ─────────────────────────────────────────────────────────────────────────────
# Vector Database
# ─────────────────────────────────────────────────────────────────────────────

def create_medical_vector_db(
    data_dir: str = MEDQUAD_DATA_DIR,
    max_docs: int = 10000,
) -> int:
    """
    Parse MedQuAD XML files, build a FAISS index, and save it locally.

    Parameters
    ----------
    data_dir : path to the MedQuAD folder
    max_docs : cap to avoid OOM on low-RAM machines (default 10 000).
               Set to 0 or None for the full 47 k dataset.

    Returns
    -------
    int  — number of documents indexed
    """
    qa_data = load_medquad_data(data_dir)

    # Apply cap
    if max_docs and len(qa_data) > max_docs:
        qa_data = qa_data[:max_docs]

    # Convert QA pairs to LangChain Documents
    documents = []
    for qa in qa_data:
        # Store both Q and A in page_content so similarity search hits either
        content = (
            f"Topic: {qa['focus']}\n"
            f"Question Type: {qa['qtype']}\n"
            f"Question: {qa['question']}\n"
            f"Answer: {qa['answer']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "focus":         qa["focus"],
                "source":        qa["source"],
                "qtype":         qa["qtype"],
                "semantic_type": qa["semantic_type"],
            }
        )
        documents.append(doc)

    # Build and save FAISS index
    vectordb = FAISS.from_documents(documents=documents, embedding=instructor_embeddings)
    vectordb.save_local(MEDICAL_VECTORDB)

    return len(documents)


def medical_vector_db_exists() -> bool:
    """Return True if the medical FAISS index has already been built."""
    return os.path.isdir(MEDICAL_VECTORDB)


# ─────────────────────────────────────────────────────────────────────────────
# QA Chain
# ─────────────────────────────────────────────────────────────────────────────

_MEDICAL_PROMPT = """You are a knowledgeable and empathetic medical information assistant \
powered by the MedQuAD dataset from the National Institutes of Health (NIH).

Use ONLY the following retrieved medical context to answer the question.
- Be clear, accurate, and concise.
- Structure the answer with bullet points where appropriate.
- Always end with: "⚕️ Please consult a qualified healthcare professional for personalised advice."
- If the answer cannot be found in the context, respond with:
  "I don't have specific information on this topic. Please consult a healthcare professional."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

MEDICAL_PROMPT = PromptTemplate(
    template=_MEDICAL_PROMPT,
    input_variables=["context", "question"]
)


def get_medical_qa_chain():
    """
    Load the saved medical FAISS index and return a RetrievalQA chain.

    Raises
    ------
    FileNotFoundError  if the index has not been built yet.
    """
    if not medical_vector_db_exists():
        raise FileNotFoundError(
            "Medical knowledge base not found. "
            "Please build it first by clicking 'Build Medical Knowledge Base'."
        )

    vectordb  = FAISS.load_local(MEDICAL_VECTORDB, instructor_embeddings)
    retriever = vectordb.as_retriever(
        score_threshold=0.6,
        search_kwargs={"k": 4}          # retrieve top-4 most relevant QA pairs
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": MEDICAL_PROMPT},
    )
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Utility — extract clean source info for display
# ─────────────────────────────────────────────────────────────────────────────

def extract_source_info(source_docs: list) -> list[dict]:
    """
    Pull display-friendly metadata out of the retrieved source documents.

    Returns list of dicts with keys: focus, source, qtype, snippet
    """
    seen    = set()
    sources = []
    for doc in source_docs:
        m     = doc.metadata
        focus = m.get("focus", "—")
        key   = (focus, m.get("qtype", ""))
        if key in seen:
            continue
        seen.add(key)
        # First 180 chars of the answer portion of page_content
        snippet = ""
        for line in doc.page_content.split("\n"):
            if line.startswith("Answer:"):
                snippet = line[7:].strip()[:180] + "…"
                break
        sources.append({
            "focus":  focus,
            "source": m.get("source", "NIH"),
            "qtype":  m.get("qtype", "—"),
            "snippet": snippet,
        })
    return sources
