import streamlit as st
import time
from typing import List

# ==========================================================
# CONFIG — TUNED FOR LARGE DOCUMENTS (40k–70k lines)
# ==========================================================

MAX_SAFE_RUNTIME = 60          # hard Streamlit execution window
LINES_PER_CHUNK = 400         # safe chunk size
AVG_SEC_PER_CHUNK = 0.35      # conservative backend estimate

# ==========================================================
# PAGE SETUP
# ==========================================================

st.set_page_config(page_title="Document Assistant", layout="wide")

st.title("📄 Document Assistant")
st.caption("Readable • Reliable • Streamlit-safe")

# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "lines": [],
    "total_lines": 0,
    "indexed_upto": 0,
    "indexing_done": False,
    "guidance_shown": False,
    "vector_store": [],
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================================
# FILE UPLOAD — SAME PAGE
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload document (PDF or TXT)",
    type=["txt", "pdf"]
)

# ==========================================================
# HELPERS (REPLACE INTERNALS LATER)
# ==========================================================

def read_document(file) -> List[str]:
    """
    Reads the FULL document into memory once.
    This avoids repeated parsing on reruns.
    """
    text = file.read().decode("utf-8", errors="ignore")
    return [line for line in text.splitlines() if line.strip()]


def index_chunk(lines: List[str]):
    """
    Backend-only indexing.
    Replace with embeddings + vector DB insert.
    """
    time.sleep(AVG_SEC_PER_CHUNK)
    st.session_state.vector_store.extend(lines)


def search_index(query: str) -> bool:
    """
    Very strict matching.
    If content is not in document, we explicitly say so.
    """
    q = query.lower()
    return any(q in line.lower() for line in st.session_state.vector_store)

# ==========================================================
# INDEXING ENGINE — FULL DOCUMENT GUARANTEED
# ==========================================================

def run_indexing():
    start = time.time()

    while st.session_state.indexed_upto < st.session_state.total_lines:

        elapsed = time.time() - start
        if elapsed > MAX_SAFE_RUNTIME:
            # HARD STOP — silent rerun
            st.rerun()

        end = min(
            st.session_state.indexed_upto + LINES_PER_CHUNK,
            st.session_state.total_lines
        )

        chunk = st.session_state.lines[
            st.session_state.indexed_upto : end
        ]

        index_chunk(chunk)
        st.session_state.indexed_upto = end

    st.session_state.indexing_done = True

# ==========================================================
# MAIN FLOW
# ==========================================================

if uploaded_file:

    # READ ONCE
    if not st.session_state.lines:
        st.session_state.lines = read_document(uploaded_file)
        st.session_state.total_lines = len(st.session_state.lines)

    estimated_time = (
        (st.session_state.total_lines / LINES_PER_CHUNK)
        * AVG_SEC_PER_CHUNK
    )

    # SINGLE FRONTEND PROGRESS BAR
    progress = st.progress(
        st.session_state.indexed_upto / max(1, st.session_state.total_lines)
    )

    # ONE-TIME GUIDANCE
    if estimated_time > MAX_SAFE_RUNTIME and not st.session_state.guidance_shown:
        st.info(
            f"""
            Large document detected (~{st.session_state.total_lines:,} lines).

            Indexing will continue safely in the background.
            No action needed from you.
            """
        )
        st.session_state.guidance_shown = True

    # INDEXING
    if not st.session_state.indexing_done:
        with st.spinner("📄 Reading and indexing document..."):
            run_indexing()
            progress.progress(
                st.session_state.indexed_upto / st.session_state.total_lines
            )

    if st.session_state.indexing_done:
        progress.progress(1.0)
        st.success("✅ Document fully indexed")

        # ==================================================
        # QUERY UI
        # ==================================================

        query = st.text_area(
            "Ask a question about the document",
            placeholder="Type your question here"
        )

        if st.button("🔍 Search") and query:
            with st.spinner("Searching..."):
                found = search_index(query)

            if not found:
                st.warning("This is not covered in the document")
            else:
                st.markdown("### ✍️ Answer")
                st.write("Relevant content found in the document.")
