import streamlit as st
from openai import OpenAI
import os
import pdfplumber
import time
import numpy as np

# ============================================================
# CONFIGURATION
# These values exist because Streamlit Cloud has hard limits
# ============================================================

MAX_FILE_SIZE_MB = 200
CHUNK_SIZE_WORDS = 600          # Balanced for recall vs speed
TOP_K = 8                       # How many chunks we feed the LLM
STREAMLIT_TIMEOUT_SECONDS = 85  # Hard platform limit
SAFE_CUTOFF_SECONDS = 60        # We stop indexing ourselves before crash

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ============================================================
# PDF TEXT VALIDATION
# Why: Many PDFs have tables / sparse layouts.
# We only check if *some* text exists, not page density.
# ============================================================

def check_pdf_format(uploaded_file):
    uploaded_file.seek(0)
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            extracted = 0
            for page in pdf.pages[:5]:
                text = page.extract_text(layout=True)
                if text:
                    extracted += len(text.strip())
            return extracted > 100
    except Exception:
        return False

# ============================================================
# PDF → CHUNK EXTRACTION WITH SAFE CUTOFF
# Why:
# - Streamlit kills long-running scripts abruptly
# - We stop indexing ourselves before that happens
# ============================================================

def extract_chunks_with_cutoff(uploaded_file):
    uploaded_file.seek(0)
    chunks = []

    start_time = time.time()
    cutoff_triggered = False

    progress = st.progress(0)
    timer_box = st.empty()

    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)

        for page_index, page in enumerate(pdf.pages, start=1):
            elapsed = time.time() - start_time

            # ⛔ SAFE STOP — DO NOT LET STREAMLIT CRASH
            if elapsed > SAFE_CUTOFF_SECONDS:
                cutoff_triggered = True
                break

            text = page.extract_text(layout=True)
            if not text:
                continue

            words = text.split()
            for i in range(0, len(words), CHUNK_SIZE_WORDS):
                chunks.append({
                    "page": page_index,
                    "text": " ".join(words[i:i + CHUNK_SIZE_WORDS])
                })

            progress.progress(page_index / total_pages)
            timer_box.info(f"⏱️ Indexing time elapsed: **{int(elapsed)} seconds**")

    return chunks, cutoff_triggered, int(time.time() - start_time)

# ============================================================
# EMBEDDINGS + SIMILARITY
# ============================================================

def embed_texts(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [np.array(e.embedding) for e in response.data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================
# RETRIEVAL (RAG CORE)
# ============================================================

def retrieve_top_chunks(query, vector_store, k=TOP_K):
    query_embedding = embed_texts([query])[0]

    scored = []
    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Assistant")
st.caption("Readable • Reliable • Streamlit-safe")

# ============================================================
# FILE UPLOAD (MAIN PAGE — NOT SIDEBAR)
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="For best results, keep documents under ~350–400 pages"
)

# ============================================================
# DOCUMENT INGESTION (RUNS ONLY ONCE)
# ============================================================

if uploaded_file and "vector_store" not in st.session_state:

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error("❌ File too large for Streamlit processing")
        st.stop()

    if not check_pdf_format(uploaded_file):
        st.warning("⚠️ Limited extractable text detected (tables or complex layout)")
        st.info("We will still attempt to process the document.")

    st.subheader("📄 Reading and indexing document")

    with st.spinner("Indexing document safely…"):
        chunks, cutoff_hit, elapsed = extract_chunks_with_cutoff(uploaded_file)

    # 🚨 If we hit cutoff, STOP CLEANLY AND ADVISE ONCE
    if cutoff_hit:
        st.warning(
            "⚠️ Indexing stopped to prevent Streamlit timeout.\n\n"
            "**Why this happens:**\n"
            "- Streamlit Cloud has a ~90 second hard execution limit\n\n"
            "**What you can do:**\n"
            "- Use PDFs under **350–400 pages**\n"
            "- Split large documents into smaller parts\n"
            "- Ask questions by specific page ranges\n"
        )
        st.stop()

    # Proceed only if indexing completed safely
    with st.spinner("Creating embeddings…"):
        embeddings = embed_texts([c["text"] for c in chunks])

    st.session_state.vector_store = [
        {
            "page": c["page"],
            "text": c["text"],
            "embedding": e
        }
        for c, e in zip(chunks, embeddings)
    ]

    st.success(f"✅ Indexed {len(chunks)} chunks in {elapsed} seconds")

# ============================================================
# QUESTION INPUT
# ============================================================

if "vector_store" in st.session_state:
    query = st.text_area(
        "Ask a question about the document",
        placeholder="Example: What does the document say about social responsibility?"
    )

    if st.button("🔍 Search") and query:

        with st.spinner("🔎 Finding relevant sections…"):
            matches = retrieve_top_chunks(query, st.session_state.vector_store)

        context = "\n\n".join(
            f"[Page {m['page']}]\n{m['text']}"
            for m in matches
        )

        with st.spinner("✍️ Generating answer…"):
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the provided document context. "
                            "Always cite page numbers. "
                            "If the answer is not present, say so clearly."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nQuestion: {query}"
                    }
                ],
                temperature=0.2
            )

        answer = response.choices[0].message.content

        # ✅ STREAMLIT-NATIVE ANSWER DISPLAY (VISIBLE IN ALL THEMES)
        with st.container(border=True):
            st.markdown("### 📝 Answer")
            st.markdown(answer)

        with st.expander("📄 Source sections used"):
            for m in matches:
                st.markdown(f"**Page {m['page']}**")
                st.text(m["text"][:400] + "…")
