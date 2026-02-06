import streamlit as st
from openai import OpenAI
import os
import pdfplumber

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI Playbook Assistant", layout="centered")

st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

# --------------------------------------------------
# SYSTEM PROMPT
# This prompt STRICTLY constrains the model behavior.
# We intentionally remove any freedom to explain itself.
# --------------------------------------------------
SYSTEM_PROMPT = """
You are a STRICT Organization Design Playbook Interpreter.

You are NOT a general consultant.
You are NOT allowed to explain what you can do.
You are NOT allowed to summarize capabilities.
You are NOT allowed to answer conversational or meta questions.

You may ONLY answer questions whose answers are explicitly present
in the provided organization design playbook.

If a question:
- is conversational (e.g. “how are you”)
- asks about your abilities
- asks for general advice
- asks anything not clearly grounded in the playbook

You MUST reply with EXACTLY this sentence and nothing else:
"This is not covered in the playbook."

You must not add explanations.
You must not soften the refusal.
You must not reframe the question.
You must not speculate.

Tone when answering valid questions:
Clear. Grounded. Slightly dry. No buzzwords.
"""

# --------------------------------------------------
# CHAPTER DEFINITIONS
# These are derived from the playbook TOC.
# We explicitly define boundaries to maintain auditability.
# --------------------------------------------------
CHAPTERS = {
    "chapter_1": {
        "title": "What is organization design?",
        "keywords": ["organization design", "structure", "systems approach"]
    },
    "chapter_2": {
        "title": "You and organization design",
        "keywords": ["roles", "capability", "skills", "designer"]
    },
    "chapter_3": {
        "title": "Finding the right sponsor",
        "keywords": ["sponsor", "ownership", "governance"]
    },
    "chapter_4": {
        "title": "Phase one – Preparing for change",
        "keywords": ["preparing", "readiness", "data gathering"]
    },
    "chapter_5": {
        "title": "Phase two – Choosing to re-design",
        "keywords": ["re-design", "scope", "vision", "principles"]
    },
    "chapter_6": {
        "title": "The communications plan",
        "keywords": ["communication", "messaging", "stakeholders"]
    },
    "chapter_7": {
        "title": "Managing stakeholders",
        "keywords": ["stakeholders", "trust", "risk"]
    },
    "chapter_8": {
        "title": "Phase three – Creating the design",
        "keywords": ["design principles", "high-level design"]
    },
    "chapter_9": {
        "title": "Risk",
        "keywords": ["risk", "mitigation", "assessment"]
    },
    "chapter_10": {
        "title": "Project management",
        "keywords": ["project", "timeline", "governance"]
    },
    "chapter_11": {
        "title": "Phase four – Handling the transition",
        "keywords": ["transition", "implementation", "change"]
    },
    "chapter_12": {
        "title": "People planning",
        "keywords": ["people", "selection", "roles"]
    },
    "chapter_13": {
        "title": "Phase five – Reviewing the design",
        "keywords": ["review", "evaluation", "PIR"]
    },
    "chapter_14": {
        "title": "Trends in organization design",
        "keywords": ["trends", "future", "new designs"]
    }
}

# --------------------------------------------------
# PDF TEXT EXTRACTION
# Responsibility: Convert uploaded PDF into raw text.
# NOTE: We are NOT chunking or grounding yet.
# That comes in the next milestone.
# --------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --------------------------------------------------
# CHAPTER TEXT SPLITTING
# NOTE: This assumes chapters appear in order in the PDF.
# This is intentional and explainable.
# --------------------------------------------------
def split_playbook_by_chapter(playbook_text):
    chapter_texts = {}
    current_chapter = None

    for line in playbook_text.splitlines():
        line_lower = line.lower()

        for chapter_id, chapter_data in CHAPTERS.items():
            if chapter_data["title"].lower() in line_lower:
                current_chapter = chapter_id
                chapter_texts[current_chapter] = ""
                break

        if current_chapter:
            chapter_texts[current_chapter] += line + "\n"

    return chapter_texts

# --------------------------------------------------
# CHAPTER SELECTION (DETERMINISTIC)
# We do NOT rely on the model to decide relevance.
# --------------------------------------------------
def select_relevant_chapters(query):
    matched_chapters = []

    for chapter_id, chapter_data in CHAPTERS.items():
        for keyword in chapter_data["keywords"]:
            if keyword in query.lower():
                matched_chapters.append(chapter_id)
                break

    return matched_chapters

# --------------------------------------------------
# BUILD GROUNDED CONTEXT
# Only selected chapters are exposed to the model.
# --------------------------------------------------
def build_grounded_context(chapter_texts, selected_chapters):
    context = ""
    for chapter_id in selected_chapters:
        context += f"\n--- {CHAPTERS[chapter_id]['title']} ---\n"
        context += chapter_texts.get(chapter_id, "")
    return context


# --------------------------------------------------
# LLM CLIENT (OpenRouter via OpenAI SDK)
# Keys are injected via Streamlit Secrets
# --------------------------------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# --------------------------------------------------
# UI: PLAYBOOK UPLOAD
# This enforces "no book = no answers"
# --------------------------------------------------
st.subheader("Upload Organization Design Playbook")
uploaded_file = st.file_uploader(
    "Upload PDF playbook (organization design only)",
    type=["pdf"]
)

# --------------------------------------------------
# USER QUESTION INPUT
# --------------------------------------------------
query = st.text_area("Your question")

# --------------------------------------------------
# HARD BLOCKED INTENTS
# These are refused BEFORE the model is called.
# This is product-level safety, not prompt-level.
# --------------------------------------------------
BLOCKED_PATTERNS = [
    "how are you",
    "what can you do",
    "what do you work on",
    "who are you",
    "your capabilities",
    "help me",
    "advise me",
    "best practice",
]

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if st.button("Run"):

    # 1️⃣ Enforce playbook upload
    if not uploaded_file:
        st.warning("Please upload the organization design playbook first.")
        st.stop()

    # Extract playbook text (not yet used by the model)
    # This is groundwork for chunking + grounding later
    playbook_text = extract_text_from_pdf(uploaded_file)

    # 2️⃣ Enforce non-empty question
    if not query.strip():
        st.warning("Ask something first.")
        st.stop()

    # 3️⃣ Hard refusal for out-of-scope intent
    if any(p in query.lower() for p in BLOCKED_PATTERNS):
        st.markdown("This is not covered in the playbook.")
        st.stop()

    # 4️⃣ Model call (still ungrounded — by design at this stage)

    # Split playbook into chapters
    chapter_texts = split_playbook_by_chapter(playbook_text)
    
    # Determine which chapters are relevant
    relevant_chapters = select_relevant_chapters(query)
    
    if not relevant_chapters:
        st.markdown("This is not covered in the playbook.")
        st.stop()
    
    # Build grounded context
    grounded_context = build_grounded_context(chapter_texts, relevant_chapters)
    
    # Final model call with HARD grounding
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
    Use ONLY the following playbook excerpts to answer.
    If the answer is not present, say:
    "This is not covered in the playbook."
    
    PLAYBOOK EXCERPTS:
    {grounded_context}
    
    QUESTION:
    {query}
    """
            }
        ],
        temperature=0.1
    )
    
    st.markdown(response.choices[0].message.content)
    
