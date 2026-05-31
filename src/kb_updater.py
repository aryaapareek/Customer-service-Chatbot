"""
kb_updater.py
Task 1: Dynamic Knowledge Base Expansion
-----------------------------------------
Provides:
  - update_vector_db_from_file()  : merges a new CSV into the existing FAISS index
  - check_and_process_pending()   : scans pending_updates/ folder and processes every CSV found
  - start_scheduler()             : launches a background thread that calls the above periodically
  - stop_scheduler()              : gracefully stops the background thread
  - is_scheduler_running()        : returns current scheduler state
  - load_update_log()             : returns the persisted update history
"""

import os
import json
import threading
from datetime import datetime

from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings

# ── Constants ────────────────────────────────────────────────────────────────
VECTORDB_PATH    = "faiss_index"
PENDING_DIR      = "pending_updates"
PROCESSED_DIR    = os.path.join(PENDING_DIR, "processed")
UPDATE_LOG_FILE  = "update_log.json"

# ── Shared embedding model (same as langchain_helper so no double-load) ──────
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)

# ── Internal scheduler state ─────────────────────────────────────────────────
_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Log helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_update_log() -> list:
    """Return the full update history list from disk."""
    if os.path.exists(UPDATE_LOG_FILE):
        with open(UPDATE_LOG_FILE, "r") as fh:
            return json.load(fh)
    return []


def _append_log(entry: dict):
    log = load_update_log()
    log.append(entry)
    with open(UPDATE_LOG_FILE, "w") as fh:
        json.dump(log, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Core update logic
# ─────────────────────────────────────────────────────────────────────────────

def update_vector_db_from_file(source_file_path: str) -> tuple[bool, str]:
    """
    Load a CSV (must have 'prompt' and 'response' columns) and merge its
    documents into the existing FAISS vector database.

    Returns
    -------
    (success: bool, message: str)
    """
    # ── Guard: FAISS index must exist ────────────────────────────────────────
    if not os.path.exists(VECTORDB_PATH):
        msg = "Knowledge base not found. Please click 'Create Knowledgebase' first."
        _append_log({
            "timestamp": _now(), "source": source_file_path,
            "status": "failed", "error": msg
        })
        return False, msg

    # ── Guard: source file must exist ────────────────────────────────────────
    if not os.path.exists(source_file_path):
        msg = f"Source file not found: {source_file_path}"
        _append_log({
            "timestamp": _now(), "source": source_file_path,
            "status": "failed", "error": msg
        })
        return False, msg

    try:
        # 1. Load new documents from CSV
        loader = CSVLoader(file_path=source_file_path, source_column="prompt")
        new_docs = loader.load()

        if not new_docs:
            msg = "The source file contained no documents."
            _append_log({
                "timestamp": _now(), "source": source_file_path,
                "status": "failed", "error": msg
            })
            return False, msg

        # 2. Load existing FAISS index
        vectordb = FAISS.load_local(VECTORDB_PATH, instructor_embeddings)

        # 3. Merge new documents into the existing index
        vectordb.add_documents(new_docs)

        # 4. Persist the updated index
        vectordb.save_local(VECTORDB_PATH)

        msg = f"Successfully added {len(new_docs)} new entries to the knowledge base."
        _append_log({
            "timestamp": _now(), "source": source_file_path,
            "docs_added": len(new_docs), "status": "success"
        })
        return True, msg

    except Exception as exc:
        msg = str(exc)
        _append_log({
            "timestamp": _now(), "source": source_file_path,
            "status": "failed", "error": msg
        })
        return False, f"Update failed: {msg}"


# ─────────────────────────────────────────────────────────────────────────────
# Pending-folder watcher
# ─────────────────────────────────────────────────────────────────────────────

def check_and_process_pending() -> list[dict]:
    """
    Scan the 'pending_updates/' folder for CSV files, process each one,
    then move it to 'pending_updates/processed/' to avoid re-processing.

    Returns a list of result dicts for UI feedback.
    """
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    results = []
    csv_files = [f for f in os.listdir(PENDING_DIR) if f.endswith(".csv")]

    for csv_file in csv_files:
        file_path = os.path.join(PENDING_DIR, csv_file)
        success, message = update_vector_db_from_file(file_path)

        # Move to processed/ regardless of outcome
        dest = os.path.join(PROCESSED_DIR, csv_file)
        # Avoid overwriting if a same-name file was processed before
        if os.path.exists(dest):
            base, ext = os.path.splitext(csv_file)
            dest = os.path.join(PROCESSED_DIR, f"{base}_{_now(safe=True)}{ext}")
        os.rename(file_path, dest)

        results.append({"file": csv_file, "success": success, "message": message})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Background scheduler
# ─────────────────────────────────────────────────────────────────────────────

def _scheduler_loop(interval_seconds: int, stop_event: threading.Event):
    """Internal loop: runs check_and_process_pending() every interval_seconds."""
    while not stop_event.is_set():
        check_and_process_pending()
        stop_event.wait(interval_seconds)   # sleep, but wakes instantly on stop


def start_scheduler(interval_hours: int = 24) -> tuple[bool, str]:
    """
    Start the background auto-update scheduler.
    It will scan pending_updates/ every `interval_hours` hours.
    """
    global _scheduler_thread, _stop_event

    if _scheduler_thread and _scheduler_thread.is_alive():
        return False, "Scheduler is already running."

    _stop_event = threading.Event()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(interval_hours * 3600, _stop_event),
        daemon=True,            # dies automatically when the Streamlit process exits
        name="KB-Auto-Updater"
    )
    _scheduler_thread.start()
    return True, f"Auto-update scheduler started — checks every {interval_hours} hour(s)."


def stop_scheduler() -> str:
    """Signal the background scheduler to stop."""
    global _stop_event
    _stop_event.set()
    return "Scheduler stopped."


def is_scheduler_running() -> bool:
    """Return True if the background scheduler thread is alive."""
    return _scheduler_thread is not None and _scheduler_thread.is_alive()


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _now(safe: bool = False) -> str:
    fmt = "%Y%m%d_%H%M%S" if safe else "%Y-%m-%d %H:%M:%S"
    return datetime.now().strftime(fmt)
