import streamlit as st
from openai import OpenAI
import os
import pdfplumber
import time
import numpy as np

# =================================================
# CONFIG
# =================================================
MAX_FILE_SIZE_MB = 200
CHUNK_SIZE_WORDS = 600
TOP_K = 8
STREAMLIT_TIMEOUT_SECONDS = 85
SAFE_WARNING_SECONDS = 60
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# =================================================
# OPENAI CLIENT
# =================================================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# =================================================
# PDF TEXT CHECK (ROBUST)
# =================================================
def check_pdf_format(uploaded_file):
    uploaded_file.seek(0)
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total = 0
            for page in pdf.pages[:5]:
                text = page.extract_text(layout=True)
                if text:
                    total += len(text.strip())
            return total > 100
    except:
        return False

# =================================================
# PDF → CHUNKS
# =================================================
def extract_chunks(uploaded_file, progress_placeholder, timer_placeholder):
    uploaded_file.seek(0)
    chunks = []
    start_time = time.time()

    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)

        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True)
            if not text:
                continue

            words = text.split()
            for i in range(0, len(words), CHUNK_SIZE_WORDS):
                chunks.append({
                    "page": idx,
                    "text": " ".join(words[i:i + CHUNK_SIZE_WORDS])
                })

            elapsed = int(time.time() - start_time)

            progress_placeholder.progress(idx / total_pages)
            timer_placeholder.info(f"⏱️ Indexing time elapsed: **{elapsed} seconds**")

            if elapsed > SAFE_WARNING_SECONDS:
                st.warning(
                    "⚠️ This document is taking longer than expected.\n\n"
                    "**Streamlit guidance:**\n"
                    "- Best results under **350–400 pages**\n"
                    "- Or split the PDF into smaller parts\n"
                    "- Or ask questions by page range\n\n"
                    "We’ll continue, but timeouts may occur."
                )

    return chunks

# =================================================
# EMBEDDINGS
# =================================================
def embed_texts(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [np.array(e.embedding) for e in response.data]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =================================================
# RETRIEVAL
# =================================================
def retrieve_chunks(query, store, k=TOP_K):
    query_emb = embed_texts([query])[0]
    scored = []

    for item in store:
        score = cosine_sim(query_emb, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:k]]

# =================================================
# STREAMLIT APP
# =================================================
st.set_page_config("Document Assistant", "📄", layout="centered")

st.title("📄 Document Assistant")
st.caption("Readable • Reliable • Streamlit-safe RAG")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# =================================================
# INGEST DOCUMENT (ONCE)
# =================================================
if uploaded_file and "vector_store" not in st.session_state:

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error("❌ File too large for processing")
        st.stop()

    if not check_pdf_format(uploaded_file):
        st.warning("⚠️ Limited extractable text detected (tables or layouts)")
        st.info("We’ll still attempt indexing.")

    st.subheader("📄 Reading and indexing document")

    progress_placeholder = st.empty()
    timer_placeholder = st.empty()

    with st.spinner("Indexing document…"):
        start = time.time()

        chunks = extract_chunks(
            uploaded_file,
            progress_placeholder,
            timer_placeholder
        )

        embeddings = embed_texts([c["text"] for c in chunks])

        st.session_state.vector_store = [
            {
                "page": c["page"],
                "text": c["text"],
                "embedding": e
            }
            for c, e in zip(chunks, embeddings)
        ]

    elapsed = int(time.time() - start)
    st.success(f"✅ Indexed {len(chunks)} chunks in {elapsed} seconds")

# =================================================
# QUERY
# =================================================
query = st.text_area("Ask a question about the document")

if st.button("🔍 Search") and query and "vector_store" in st.session_state:

    with st.spinner("🔎 Retrieving relevant sections…"):
        matches = retrieve_chunks(query, st.session_state.vector_store)

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
                        "If not found, say so."
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

    with st.container(border=True):
        st.markdown("### 📝 Answer")
        st.markdown(answer)

    with st.expander("📄 Source sections"):
        for m in matches:
            st.markdown(f"**Page {m['page']}**")
            st.text(m["text"][:400] + "…")
