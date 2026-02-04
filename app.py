import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="AI Playbook Assistant", layout="centered")

st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

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


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

query = st.text_area("Your question")

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

if st.button("Run"):
    if not query.strip():
        st.warning("Ask something first.")
    elif any(p in query.lower() for p in BLOCKED_PATTERNS):
        st.markdown("This is not covered in the playbook.")
    else:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.2
        )

        st.markdown(response.choices[0].message.content)
