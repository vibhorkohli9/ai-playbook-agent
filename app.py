import streamlit as st
import time
from typing import List

# ==============================
# CONFIGURATION
# ==============================

MAX_SAFE_RUNTIME = 60  # seconds (Streamlit-safe window)
AVG_SECONDS_PER_PAGE = 0.4  # conservative estimate
CHUNK_PAGES = 5  # pages indexed per iteration

# ==============================
# STREAMLIT PAGE SETUP
# ==============================

st.set_page_config(
    page_title="Document Assistant",
    layout="wide"
)

st.title("📄 Document Assistant")
st.caption("Readable • Reliable • Streamlit-safe")

# ==============================
# SESSION STATE INIT
# ==============================

if "indexed_pages" not in st.session_state:
    st.session_state.indexed_pages = 0

if "indexing_done" not in st.session_state:
    st.session_state.indexing_done = False

if "showed_guidance" not in st.session_state:
    st.session_state.showed_guidance = False

if "doc_pages" not in st.session_state:
    st.session_state.doc_pages = 0

if "index_start_time" not in st.session_state:
    st.session_state.index_start_time = None

# ==============================
# DOCUMENT UPLOAD (SAME PAGE)
# ==============================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"],
    help="For best performance, keep documents under 150 pages"
)

# ==============================
# MOCK HELPERS (REPLACE WITH YOUR REAL LOGIC)
# ==============================

def extract_pages(file) -> List[str]:
    """
    Replace this with real PDF/text parsing.
    Returns list of pages.
    """
    # Simulated pages
    return ["Page content"] * 120


def index_pages(pages: List[str]):
    """
    Replace with:
    - embeddings
    - vector store insert
    - metadata storage
    """
    time.sleep(AVG_SECONDS_PER_PAGE * len(pages))


# ==============================
# TIME ESTIMATION LOGIC
# ==============================

def estimate_total_time(pages: int) -> float:
    """
    Estimate total indexing time BEFORE starting.
    This avoids surprising the user mid-way.
    """
    return pages * AVG_SECONDS_PER_PAGE


# ==============================
# INDEXING CONTROLLER
# ==============================

def run_indexing(pages: List[str]):
    """
    This function safely indexes documents without breaking Streamlit.
    It intentionally stops at ~60s and reruns the app.
    """

    if st.session_state.index_start_time is None:
        st.session_state.index_start_time = time.time()

    start_loop = time.time()

    while st.session_state.indexed_pages < len(pages):

        elapsed = time.time() - start_loop
        if elapsed > MAX_SAFE_RUNTIME:
            # HARD STOP: Streamlit-safe exit
            st.rerun()

        next_chunk = pages[
            st.session_state.indexed_pages :
            st.session_state.indexed_pages + CHUNK_PAGES
        ]

        index_pages(next_chunk)
        st.session_state.indexed_pages += len(next_chunk)

        progress = st.session_state.indexed_pages / len(pages)
        st.progress(progress)

    st.session_state.indexing_done = True


# ==============================
# MAIN FLOW
# ==============================

if uploaded_file:

    pages = extract_pages(uploaded_file)
    st.session_state.doc_pages = len(pages)

    estimated_time = estimate_total_time(len(pages))

    st.info(f"📊 Estimated indexing time: ~{int(estimated_time)} seconds")

    # Show guidance ONLY ONCE
    if estimated_time > MAX_SAFE_RUNTIME and not st.session_state.showed_guidance:
        st.warning(
            """
            ⏱️ This document is large and may exceed Streamlit's safe execution window.

            **What you can do:**
            - Split the document into smaller parts (50–100 pages)
            - Upload only relevant sections
            - Prefer text-based PDFs over scanned files

            👉 Indexing will continue automatically in the background.
            """
        )
        st.session_state.showed_guidance = True

    if not st.session_state.indexing_done:
        with st.spinner("📄 Reading and indexing document..."):
            run_indexing(pages)

    if st.session_state.indexing_done:
        st.success("✅ Document fully indexed and ready!")
        st.caption("You can now ask questions below.")

        # ==============================
        # QUESTION ANSWERING UI
        # ==============================

        query = st.text_area(
            "Ask a question about the document",
            placeholder="e.g. What are the key approaches to social responsibility?"
        )

        if st.button("🔍 Search") and query:
            with st.spinner("Thinking..."):
                # Replace with RAG answer logic
                time.sleep(1)
                st.markdown("### ✍️ Answer")
                st.write(
                    "This document outlines several approaches to social responsibility, "
                    "including employee welfare, community engagement, and shareholder accountability."
                )
