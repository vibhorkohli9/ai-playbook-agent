import streamlit as st
from openai import OpenAI
import os
import pdfplumber
import time


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
# BLOCK-LEVEL TEXT EXTRACTION (2000+ pages safe)
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

            # ---- Progress update
            if progress_bar:
                progress = int((page_num / total_pages) * 100)
                progress_bar.progress(progress)

    return blocks


# =================================================
# CONFIDENCE BADGE
# =================================================
def calculate_confidence(relevant_blocks):
    count = len(relevant_blocks)

    if count >= 8:
        return "🟢🟢 Confidence: High (multiple strong evidence blocks)"
    elif count >= 3:
        return "🟢 Confidence: Medium (clear but limited evidence)"
    else:
        return "🟡 Confidence: Low (narrow reference)"


# =================================================
# STREAMLIT CONFIG
# =================================================
st.set_page_config(page_title="AI Document Assistant", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    textarea, input { background-color: #1E222A !important; color: #FAFAFA !important; }
    button[kind="primary"] { background-color: #4F8BF9; color: white; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

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
    st.header("📄 Document Control")
    uploaded_file = st.file_uploader("Upload any text-based PDF", type=["pdf"])

    st.markdown("---")
    st.caption("⚠️ Trust Guarantees")
    st.caption("• Evidence-only answers")
    st.caption("• No hallucinations")
    st.caption("• Mandatory citations")


# =================================================
# QUESTION
# =================================================
query = st.text_area(
    "Ask a grounded question",
    placeholder="e.g. What governance model is recommended?"
)


# =================================================
# RUN
# =================================================
if st.button("Run"):

    if not uploaded_file:
        st.warning("Please upload a document.")
        st.stop()

    if not document_suitability_check(uploaded_file):
        st.error("⚠️ Scanned or image-based PDF detected.")
        st.stop()

    if not query.strip():
        st.warning("Please ask a question.")
        st.stop()

    # ---- Loading UI
    st.markdown("### ⏳ Processing document")
    progress_bar = st.progress(0)
    status = st.empty()

    status.markdown("📖 Reading pages…")

    blocks = extract_text_blocks(uploaded_file, progress_bar=progress_bar)

    status.markdown("🔎 Finding evidence…")

    keywords = query.lower().split()
    relevant_blocks = [
        b for b in blocks
        if any(k in b["text"].lower() for k in keywords)
    ]

    if not relevant_blocks:
        st.markdown("This is not covered in the document.")
        st.stop()

    confidence_badge = calculate_confidence(relevant_blocks)

    context_blocks = []
    for b in relevant_blocks[:10]:
        context_blocks.append(
            f"""
Block ID: {b['id']}
Page: {b['page']}
Content:
{b['text']}
"""
        )

    context_text = "\n\n".join(context_blocks)

    status.markdown("🧠 Generating answer…")
    progress_bar.progress(100)

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

    status.empty()
    progress_bar.empty()

    # =================================================
    # OUTPUT
    # =================================================
    st.markdown("### ✅ Grounded Answer")
    st.markdown(response.choices[0].message.content)

    with st.expander("📚 Evidence used from the document"):
        for b in relevant_blocks[:10]:
            st.markdown(f"**Block ID:** {b['id']}  \n**Page:** {b['page']}")
            st.markdown(b["text"][:400] + "…")
            st.markdown("---")

    st.markdown("---")
    st.markdown(confidence_badge)
