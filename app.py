import streamlit as st
from openai import OpenAI
import os
import pdfplumber


# =================================================
# DOCUMENT SUITABILITY CHECK
# =================================================
def document_suitability_check(uploaded_file, sample_pages=10):
    uploaded_file.seek(0)

    with pdfplumber.open(uploaded_file) as pdf:
        text_pages = 0
        for page in pdf.pages[:sample_pages]:
            if page.extract_text():
                text_pages += 1

    return text_pages >= max(1, sample_pages // 3)


# =================================================
# BLOCK-LEVEL TEXT EXTRACTION
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
# CONFIDENCE BADGE
# =================================================
def calculate_confidence(block_count):
    if block_count >= 8:
        return "🟢🟢 Confidence: High (multiple strong evidence blocks)"
    elif block_count >= 3:
        return "🟢 Confidence: Medium (clear but limited evidence)"
    else:
        return "🟡 Confidence: Low (narrow reference)"


# =================================================
# STREAMLIT CONFIG
# =================================================
st.set_page_config(page_title="AI Document Assistant", layout="centered")

st.title("AI Document Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")


# =================================================
# SYSTEM PROMPT
# =================================================
SYSTEM_PROMPT = """
You are a STRICT Document Evidence Interpreter.

Rules:
- Answer ONLY from provided text
- Every answer MUST include citations
- If not found, reply exactly:
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
# QUESTION
# =================================================
query = st.text_area("Ask a grounded question")


# =================================================
# RUN
# =================================================
if st.button("Run"):

    if not uploaded_file or not query.strip():
        st.warning("Upload a document and ask a question.")
        st.stop()

    if not document_suitability_check(uploaded_file):
        st.error("⚠️ Scanned or image-based PDF detected.")
        st.stop()

    st.markdown("### ⏳ Processing document")
    progress_bar = st.progress(0)

    blocks = extract_text_blocks(uploaded_file, progress_bar=progress_bar)

    keywords = query.lower().split()
    relevant_blocks = [
        b for b in blocks
        if any(k in b["text"].lower() for k in keywords)
    ]

    if not relevant_blocks:
        st.markdown("This is not covered in the document.")
        st.stop()

    context_text = "\n\n".join(
        f"Block ID: {b['id']}\nPage: {b['page']}\nContent:\n{b['text']}"
        for b in relevant_blocks[:10]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Document excerpts:\n{context_text}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()

    # =================================================
    # TRUST GATE (CRITICAL FIX)
    # =================================================
    st.markdown("### ✅ Grounded Answer")
    st.markdown(answer)

    if answer == "This is not covered in the document.":
        # STOP — no evidence, no confidence
        st.stop()

    # =================================================
    # Evidence Preview
    # =================================================
    with st.expander("📚 Evidence used from the document"):
        for b in relevant_blocks[:10]:
            st.markdown(f"**Block ID:** {b['id']} | **Page:** {b['page']}")
            st.markdown(b["text"][:400] + "…")
            st.markdown("---")

    # =================================================
    # Confidence (ONLY FOR VALID ANSWERS)
    # =================================================
    st.markdown("---")
    st.markdown(calculate_confidence(len(relevant_blocks)))
