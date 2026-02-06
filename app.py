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
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )

    st.markdown(response.choices[0].message.content)
