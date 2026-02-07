import streamlit as st
from openai import OpenAI
import os
import pdfplumber
import re

# -------------------------------------------------
# Streamlit UI config
# -------------------------------------------------
st.set_page_config(page_title="AI Playbook Assistant", layout="centered")

st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

# -------------------------------------------------
# SYSTEM PROMPT — now enforces chapter citation
# -------------------------------------------------
SYSTEM_PROMPT = """
You are a STRICT Organization Design Playbook Interpreter.

Rules you MUST follow:
- You may ONLY answer using the provided playbook excerpts.
- Every valid answer MUST include a source citation.
- Citation format MUST be:
  Chapter: <chapter name>
  Pages: <page numbers>

If the answer is not explicitly present in the provided text,
reply with EXACTLY this sentence and nothing else:
"This is not covered in the playbook."

You must not:
- Explain your capabilities
- Add external knowledge
- Speculate
- Soften refusals
"""

# -------------------------------------------------
# Utility: Extract chapters from PDF
# -------------------------------------------------
def extract_chapters_from_pdf(uploaded_file):
    """
    Reads PDF page by page.
    Attempts to detect chapter headings.
    Returns a list of chapter chunks with metadata.
    """

    chapters = []
    current_chapter = {
        "title": "Unknown Section",
        "pages": [],
        "text": ""
    }

    chapter_heading_pattern = re.compile(
        r"^(chapter\s+\d+|section\s+\d+|[A-Z][A-Za-z\s]{5,})",
        re.IGNORECASE
    )

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
def calculate_confidence(relevant_chapters):
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

    chapter_count = len(relevant_chapters)

    if chapter_count >= 3 and total_pages >= 6:
        return "🟢🟢 Confidence: High (multiple chapters, broad coverage)"
    elif chapter_count >= 1 and total_pages >= 3:
        return "🟢 Confidence: Medium (limited but clear coverage)"
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
st.subheader("Upload Organization Design Playbook")

uploaded_file = st.file_uploader(
    "Upload PDF playbook (organization design only)",
    type=["pdf"]
)

query = st.text_area("Your question")

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

# -------------------------------------------------
# Main execution
# -------------------------------------------------
if st.button("Run"):

    if not uploaded_file:
        st.warning("Please upload the organization design playbook first.")
        st.stop()

    if not query.strip():
        st.warning("Ask something first.")
        st.stop()

    if any(p in query.lower() for p in BLOCKED_PATTERNS):
        st.markdown("This is not covered in the playbook.")
        st.stop()

    # Step 1: Extract chapters
    chapters = extract_chapters_from_pdf(uploaded_file)

    # Step 2: Filter relevant chapters
    relevant_chapters = filter_relevant_chapters(chapters, query)
    relevant_chapters = relevant_chapters or []

    # -------------------------------------------------
    # HARD GROUNDING GATE
    # If no chapter contains evidence for the query,
    # the model must NEVER be called.
    # -------------------------------------------------
    if not relevant_chapters:
        st.markdown("This is not covered in the playbook.")
        st.stop()  

    # Step 2.5: Calculate confidence from grounding
    confidence_badge = calculate_confidence(relevant_chapters)

  
    # Step 3: Build grounded context
    context_blocks = []
    for ch in relevant_chapters:
        context_blocks.append(
            f"""
Chapter: {ch['title']}
Pages: {min(ch['pages'])}-{max(ch['pages'])}
Content:
{ch['text'][:3000]}
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

    st.markdown(response.choices[0].message.content)

    st.markdown("---")
    st.markdown(confidence_badge)
