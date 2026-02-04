import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="AI Playbook Assistant", layout="centered")

st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

SYSTEM_PROMPT = """
You are an Organization Design Consultant AI.

Rules you MUST follow:
- You ONLY answer using the provided playbook content.
- If the answer is not explicitly present in the playbook, say:
  "This is not covered in the playbook."
- Do NOT use external knowledge.
- Do NOT hallucinate.
- Be direct, grounded, and practical.
- Tone: intelligent, slightly dry, no buzzwords.
"""

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

query = st.text_area("Your question")

if st.button("Run"):
    if not query.strip():
        st.warning("Ask something first.")
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
