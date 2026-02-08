import streamlit as st
from openai import OpenAI
import os
import pdfplumber
import time
import numpy as np

# =================================================
# CONFIGURATION
# =================================================
MAX_FILE_SIZE_MB = 200
CHUNK_SIZE_WORDS = 600
TOP_K = 8
STREAMLIT_TIMEOUT_SECONDS = 85
PROCESSING_SPEED_PAGES_PER_SECOND = 5
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# =================================================
# UI CSS (FIXED VISIBILITY)
# =================================================
def inject_custom_css():
    st.markdown("""
    <style>
    .time-estimate, .warning-box {
        color: #1F2937 !important;
        font-weight: 600;
    }

    .time-estimate {
        background: #FFF7CC;
        border-left: 6px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
    }

    .warning-box {
        background: #FEE2E2;
        border-left: 6px solid #DC2626;
        padding: 1.5rem;
        border-radius: 8px;
    }

    .answer-box {
        background: white;
        border-left: 4px solid #DC2626;
        padding: 1.5rem;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================
# OPENAI CLIENT
# =================================================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# =================================================
# PDF TEXT CHECK (FIXED)
# =================================================
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
    except:
        return False

# =================================================
# PDF → CHUNKS
# =================================================
def extract_chunks(uploaded_file):
    uploaded_file.seek(0)
    chunks = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True)
            if not text:
                continue

            words = text.split()
            for i in range(0, len(words), CHUNK_SIZE_WORDS):
                chunk = " ".join(words[i:i+CHUNK_SIZE_WORDS])
                chunks.append({
                    "page": page_no,
                    "text": chunk
                })
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
# RAG RETRIEVAL
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
st.set_page_config("Document Assistant (RAG)", "📄")
inject_custom_css()

st.title("📄 Document Assistant (RAG)")
st.caption("Semantic search across your entire document")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# =================================================
# INGEST DOCUMENT ONCE
# =================================================
if uploaded_file and "vector_store" not in st.session_state:

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error("File too large")
        st.stop()

    if not check_pdf_format(uploaded_file):
        st.warning("⚠️ Limited extractable text detected")
        st.info("💡 Document may contain tables or complex layouts. Processing anyway.")

    with st.spinner("📄 Reading and indexing document..."):
        start = time.time()
        chunks = extract_chunks(uploaded_file)

        embeddings = embed_texts([c["text"] for c in chunks])

        st.session_state.vector_store = [
            {
                "page": c["page"],
                "text": c["text"],
                "embedding": e
            }
            for c, e in zip(chunks, embeddings)
        ]

    st.success(f"Indexed {len(chunks)} chunks in {time.time() - start:.1f}s")

# =================================================
# QUERY
# =================================================
query = st.text_area("Ask a question about the document")

if st.button("🔍 Search") and query and "vector_store" in st.session_state:

    with st.spinner("🔎 Retrieving relevant sections..."):
        matches = retrieve_chunks(query, st.session_state.vector_store)

    context = "\n\n".join(
        f"[Page {m['page']}]\n{m['text']}"
        for m in matches
    )

    with st.spinner("✍️ Generating answer..."):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a document assistant.
Answer ONLY from provided content.
Always cite page numbers.
If not found, say so."""
                },
                {
                    "role": "user",
                    "content": f"""
Document context:
{context}

Question: {query}
"""
                }
            ],
            temperature=0.2
        )

    answer = response.choices[0].message.content

    st.markdown("### 📝 Answer")
    st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

    with st.expander("📄 Source Sections"):
        for m in matches:
            st.markdown(f"**Page {m['page']}**")
            st.text(m["text"][:400] + "...")
