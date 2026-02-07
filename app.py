import streamlit as st
from openai import OpenAI
import os
import pdfplumber
import re

def document_suitability_check(uploaded_file):
    """
    Warns if the PDF is likely scanned / image-based.
    """
    with pdfplumber.open(uploaded_file) as pdf:
        text_pages = 0
        for page in pdf.pages[:10]:  # sample first 10 pages
            if page.extract_text():
                text_pages += 1

    if text_pages < 3:
        return False
    return True


# -------------------------------------------------
# Streamlit UI config
# -------------------------------------------------
st.set_page_config(page_title="AI Playbook Assistant", layout="centered")
# -------------------------------------------------
# UI Theme Styling
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Text inputs & text areas */
    textarea, input {
        background-color: #1E222A !important;
        color: #FAFAFA !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #4F8BF9;
        color: white;
        border-radius: 8px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

# -------------------------------------------------
# SYSTEM PROMPT — now enforces chapter citation
# -------------------------------------------------
SYSTEM_PROMPT = """
You are a STRICT Document Evidence Interpreter.

Rules you MUST follow:
- You may ONLY answer using the provided document excerpts.
- Every valid answer MUST include a source citation.
- Citation format MUST be:
  Section: <section title or block id>
  Pages: <page numbers>

If the answer is not explicitly present in the provided text,
reply with EXACTLY this sentence and nothing else:
"This is not covered in the document."

You must not:
- Explain your capabilities
- Add external knowledge
- Speculate
- Soften refusals
"""


# -------------------------------------------------
# Utility: Extract chapters from PDF
# -------------------------------------------------
def extract_text_blocks(uploaded_file, block_size=800):
    """
    Extracts small text blocks per page.
    Safe for 2000+ page documents.
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
                chunk = " ".join(words[i:i + block_size])
                blocks.append({
                    "id": f"B{block_id}",
                    "page": page_num,
                    "text": chunk
                })
                block_id += 1

    return blocks



    # -------------------------------------------------
# Utility: Check if PDF is text-based (not scanned)
# -------------------------------------------------
def is_text_based_pdf(uploaded_file, sample_pages=5):
    """
    Checks whether the PDF contains extractable text.
    Prevents silent failures on scanned/image PDFs.
    """

    with pdfplumber.open(uploaded_file) as pdf:
        text_pages = 0

        for page in pdf.pages[:sample_pages]:
            if page.extract_text():
                text_pages += 1

    # Require text on at least half the sampled pages
    return text_pages >= max(1, sample_pages // 2)

    
    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""

            lines = page_text.split("\n")

            for line in lines:
                # Detect chapter-like headings
                if chapter_heading_pattern.match(line.strip()):
                    # Save previous chapter
                    if current_chapter["text"].strip():
                        chapters.append(current_chapter)

                    # Start new chapter
                    current_chapter = {
                        "title": line.strip(),
                        "pages": [page_num],
                        "text": ""
                    }
                else:
                    current_chapter["text"] += line + " "

            if page_num not in current_chapter["pages"]:
                current_chapter["pages"].append(page_num)

    # Append last chapter
    if current_chapter["text"].strip():
        chapters.append(current_chapter)

    return chapters

# -------------------------------------------------
# Utility: Select relevant chapters (simple keyword filter)
# -------------------------------------------------
def filter_relevant_chapters(chapters, query):
    """
    Naive but effective Phase-2A filtering.
    Only chapters containing query keywords are passed to the model.
    """

    keywords = query.lower().split()
    relevant = []

    for ch in chapters:
        text_lower = ch["text"].lower()
        if any(k in text_lower for k in keywords):
            relevant.append(ch)

    return relevant

# -------------------------------------------------
# Utility: Confidence badge (Phase-2A, rule-based)
# -------------------------------------------------
def calculate_confidence(relevant_blocks):
    """
    Confidence is derived ONLY from grounding strength.
    No ML, no guessing.
    """

    if not relevant_chapters:
        return "❌ Confidence: No evidence found in playbook"

    total_pages = sum(
        max(ch["pages"]) - min(ch["pages"]) + 1
        for ch in relevant_chapters
    )

    block_count = len(relevant_blocks)

    if block_count >= 8:
        return "🟢🟢 Confidence: High (multiple evidence blocks)"
    elif block_count >= 3:
        return "🟢 Confidence: Medium (clear but limited evidence)"
    else:
        return "🟡 Confidence: Low (narrow reference)"


# -------------------------------------------------
# OpenRouter / OpenAI-compatible client
# -------------------------------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# -------------------------------------------------
# UI — File upload
# -------------------------------------------------
with st.sidebar:
    st.header("📘 Playbook Control")

    uploaded_file = st.file_uploader(
        "Upload Organization Design Playbook (PDF)",
        type=["pdf"]
    )

    st.markdown("---")
    st.caption("⚠️ Grounding Rules")
    st.caption("• Answers only from the playbook")
    st.caption("• No speculation")
    st.caption("• Chapter + page citation mandatory")


query = st.text_area(
    "Ask a grounded question",
    placeholder="e.g. How should decision rights be structured in a matrix organization?"
)
st.caption("Only questions answerable from the uploaded playbook will return results.")


# Explicit conversational blockers
BLOCKED_PATTERNS = [
    "how are you",
    "what can you do",
    "who are you",
    "your capabilities",
    "help me",
    "advise me",
    "best practice",
]

st.markdown("### 🔎 Grounding Status")
st.markdown("🧠 Model access is locked until textual evidence is found.")

# -------------------------------------------------
# Main execution
# -------------------------------------------------
if st.button("Run"):

    if not uploaded_file:
        st.warning("Please upload the organization design playbook first.")
        st.stop()

    # Document suitability check
    if not document_suitability_check(uploaded_file):
    st.error(
        "⚠️ This document appears to be scanned or image-based.\n\n"
        "Please upload a text-based PDF for reliable grounded answers."
    )
    st.stop()



    if not query.strip():
        st.warning("Ask something first.")
        st.stop()

    if any(p in query.lower() for p in BLOCKED_PATTERNS):
        st.markdown("This is not covered in the playbook.")
        st.stop()

    blocks = extract_text_blocks(uploaded_file)

    keywords = query.lower().split()
    relevant_blocks = [
        b for b in blocks
        if any(k in b["text"].lower() for k in keywords)
    ]

    
    # -------------------------------------------------
    # HARD GROUNDING GATE
    # If no chapter contains evidence for the query,
    # the model must NEVER be called.
    # -------------------------------------------------
    if not relevant_blocks:
        st.markdown("This is not covered in the playbook.")
        st.stop()  

    # Step 2.5: Calculate confidence from grounding
    confidence_badge = calculate_confidence(relevant_chapters)

  
    # Step 3: Build grounded context
    context_blocks = []
    for b in relevant_blocks[:10]:  # HARD LIMIT = token safety
        context_blocks.append(
            f"""
    Block ID: {b['id']}
    Page: {b['page']}
    Content:
    {b['text']}
    """
        )
    
    context_text = "\n\n".join(context_blocks)


  
    # Step 4: Call model
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {
              "role": "user",
              "content": f"""
  Playbook excerpts:
  {context_text}
  
  Question:
  {query}
  """
          }
      ],
      temperature=0.2
  )

    # -------------------------------------------------
    # Display Grounded Answer
    # -------------------------------------------------
    st.markdown("### ✅ Grounded Answer")
    st.markdown(response.choices[0].message.content)
    
    # -------------------------------------------------
    # Evidence Preview (AFTER answer, BEFORE confidence)
    # -------------------------------------------------
    with st.expander("📚 Evidence used from the document"):
    for b in relevant_blocks[:10]:
        st.markdown(
            f"**Block ID:** {b['id']}  \n"
            f"**Page:** {b['page']}"
        )
        st.markdown(b["text"][:400] + "...")
        st.markdown("---")
    
    # -------------------------------------------------
    # Confidence Badge (LAST)
    # -------------------------------------------------
    st.markdown("---")
    st.markdown(confidence_badge)

