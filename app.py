import streamlit as st
from openai import OpenAI
import os
import pdfplumber


# =================================================
# PDF TEXT CHECK
# =================================================
def document_suitability_check(uploaded_file, sample_pages=10):
    uploaded_file.seek(0)

    with pdfplumber.open(uploaded_file) as pdf:
        text_pages = sum(
            1 for p in pdf.pages[:sample_pages] if p.extract_text()
        )

    return text_pages >= max(1, sample_pages // 3)


# =================================================
# BLOCK EXTRACTION (LARGE PDFs SAFE)
# =================================================
def extract_text_blocks(uploaded_file, block_size=800, progress_bar=None):
    uploaded_file.seek(0)
    blocks = []
    block_id = 1

    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            words = text.split()
            for i in range(0, len(words), block_size):
                blocks.append({
                    "id": f"B{block_id}",
                    "page": page_num,
                    "text": " ".join(words[i:i + block_size])
                })
                block_id += 1

            if progress_bar:
                progress_bar.progress(int((page_num / total_pages) * 100))

    return blocks


# =================================================
# SOFT RELEVANCE SCORING (CRITICAL FIX)
# =================================================
def score_blocks(blocks, query):
    keywords = set(query.lower().split())

    scored = []
    for b in blocks:
        text = b["text"].lower()
        score = sum(1 for k in keywords if k in text)
        scored.append((score, b))

    # Highest score first
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return blocks only
    return [b for _, b in scored]


# =================================================
# CONFIDENCE BADGE
# =================================================
def calculate_confidence(block_count):
    if block_count >= 8:
        return "🟢🟢 Confidence: High"
    elif block_count >= 3:
        return "🟢 Confidence: Medium"
    else:
        return "🟡 Confidence: Low"


# =================================================
# STREAMLIT SETUP
# =================================================
st.set_page_config(page_title="AI Document Assistant", layout="centered")

st.title("AI Document Assistant")
st.caption("Ask a question. Get an answer strictly from the document.")


# =================================================
# SYSTEM PROMPT
# =================================================
SYSTEM_PROMPT = """
You are a STRICT Document Evidence Interpreter.

Rules:
- Answer ONLY using the provided document excerpts
- Every answer MUST include citations
- If the answer is not explicitly present, reply EXACTLY:
"This is not covered in the document."
"""


# =================================================
# OPENAI CLIENT
# =================================================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a text-based PDF", type=["pdf"])


# =================================================
# USER QUESTION
# =================================================
query = st.text_area("Ask a question from the document")


# =================================================
# RUN
# =================================================
if st.button("Run"):

    if not uploaded_file or not query.strip():
        st.warning("Upload a document and enter a question.")
        st.stop()

    if not document_suitability_check(uploaded_file):
        st.error("⚠️ This PDF appears to be scanned or image-based.")
        st.stop()

    st.markdown("### ⏳ Processing document")
    progress = st.progress(0)

    blocks = extract_text_blocks(uploaded_file, progress_bar=progress)

    # 🔥 KEY CHANGE: NO HARD FILTER
    ranked_blocks = score_blocks(blocks, query)

    # Always pass top blocks to model
    context_blocks = ranked_blocks[:12]

    context_text = "\n\n".join(
        f"Block ID: {b['id']}\nPage: {b['page']}\nContent:\n{b['text']}"
        for b in context_blocks
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
Document excerpts:
{context_text}

Question:
{query}
"""
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()

    # =================================================
    # OUTPUT (TRUST SAFE)
    # =================================================
    st.markdown("### ✅ Answer")
    st.markdown(answer)

    if answer == "This is not covered in the document.":
        st.stop()

    with st.expander("📚 Evidence from document"):
        for b in context_blocks:
            st.markdown(f"**Block ID:** {b['id']} | **Page:** {b['page']}")
            st.markdown(b["text"][:400] + "…")
            st.markdown("---")

    st.markdown("---")
    st.markdown(calculate_confidence(len(context_blocks)))
