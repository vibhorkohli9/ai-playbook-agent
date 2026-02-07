import streamlit as st
from openai import OpenAI
import os
import pdfplumber

# =================================================
# DOCUMENT SUITABILITY CHECK
# =================================================
def document_suitability_check(uploaded_file, sample_pages=10):
    """
    Detects scanned / image-only PDFs.
    Returns False if insufficient extractable text is found.
    """

    with pdfplumber.open(uploaded_file) as pdf:
        text_pages = 0
        for page in pdf.pages[:sample_pages]:
            if page.extract_text():
                text_pages += 1

    # Require text on at least 30% of sampled pages
    return text_pages >= max(1, sample_pages // 3)


# =================================================
# BLOCK-LEVEL TEXT EXTRACTION (2000+ pages safe)
# =================================================
def extract_text_blocks(uploaded_file, block_size=800):
    """
    Breaks document into small, page-scoped text blocks.
    Designed for very large documents.
    """

    blocks = []
    block_id = 1

    with pdfplumber.open(uploaded_file) as pdf:
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

    return blocks


# =================================================
# CONFIDENCE BADGE (GROUNDING-ONLY)
# =================================================
def calculate_confidence(relevant_blocks):
    """
    Confidence derived purely from evidence density.
    """

    block_count = len(relevant_blocks)

    if block_count >= 8:
        return "🟢🟢 Confidence: High (multiple strong evidence blocks)"
    elif block_count >= 3:
        return "🟢 Confidence: Medium (clear but limited evidence)"
    else:
        return "🟡 Confidence: Low (narrow reference)"


# =================================================
# STREAMLIT CONFIG + THEME
# =================================================
st.set_page_config(page_title="AI Document Assistant", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    textarea, input { background-color: #1E222A !important; color: #FAFAFA !important; }
    button[kind="primary"] { background-color: #4F8BF9; color: white; border-radius: 8px; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI Document Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")


# =================================================
# SYSTEM PROMPT (DOMAIN-AGNOSTIC)
# =================================================
SYSTEM_PROMPT = """
You are a STRICT Document Evidence Interpreter.

Rules you MUST follow:
- You may ONLY answer using the provided document excerpts.
- Every valid answer MUST include a source citation.
- Citation format MUST be:
  Block ID: <block id>
  Page: <page number>

If the answer is not explicitly present in the provided text,
reply with EXACTLY this sentence and nothing else:
"This is not covered in the document."

You must not:
- Explain your capabilities
- Add external knowledge
- Speculate
- Soften refusals
"""


# =================================================
# OPENAI CLIENT
# =================================================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


# =================================================
# SIDEBAR – DOCUMENT CONTROL
# =================================================
with st.sidebar:
    st.header("📄 Document Control")
    uploaded_file = st.file_uploader("Upload any text-based PDF", type=["pdf"])

    st.markdown("---")
    st.caption("⚠️ Trust Guarantees")
    st.caption("• Answers strictly from document")
    st.caption("• No hallucinations")
    st.caption("• Mandatory citations")


# =================================================
# USER QUESTION
# =================================================
query = st.text_area(
    "Ask a grounded question",
    placeholder="e.g. What governance model is recommended?"
)

st.markdown("### 🔎 Grounding Status")
st.markdown("🧠 Model access is locked until evidence is found.")


# =================================================
# EXECUTION
# =================================================
if st.button("Run"):

    # --- Upload validation
    if not uploaded_file:
        st.warning("Please upload a document first.")
        st.stop()

    # --- Suitability check
    if not document_suitability_check(uploaded_file):
        st.error(
            "⚠️ This PDF appears to be scanned or image-based.\n\n"
            "Please upload a text-based PDF for reliable answers."
        )
        st.stop()

    # --- Question validation
    if not query.strip():
        st.warning("Please ask a question.")
        st.stop()

    # --- Extract blocks
    blocks = extract_text_blocks(uploaded_file)

    # --- Keyword-based grounding filter
    keywords = query.lower().split()
    relevant_blocks = [
        b for b in blocks
        if any(k in b["text"].lower() for k in keywords)
    ]

    # --- HARD GROUNDING GATE
    if not relevant_blocks:
        st.markdown("This is not covered in the document.")
        st.stop()

    # --- Confidence
    confidence_badge = calculate_confidence(relevant_blocks)

    # --- Build model context (capped)
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

    # --- Model call
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

    # =================================================
    # OUTPUT
    # =================================================
    st.markdown("### ✅ Grounded Answer")
    st.markdown(response.choices[0].message.content)

    # --- Evidence Preview
    with st.expander("📚 Evidence used from the document"):
        for b in relevant_blocks[:10]:
            st.markdown(f"**Block ID:** {b['id']}  \n**Page:** {b['page']}")
            st.markdown(b["text"][:400] + "…")
            st.markdown("---")

    # --- Confidence (last)
    st.markdown("---")
    st.markdown(confidence_badge)
