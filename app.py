import streamlit as st
from openai import OpenAI
import os

system_prompt = """
You are an Organization Design Consultant AI.

Rules you MUST follow:
- You ONLY answer using the provided playbook content.
- If the answer is not explicitly present in the playbook, say:
  "This is not covered in the playbook."
- Do NOT use external knowledge.
- Do NOT hallucinate.
- Be direct, grounded, and practical.
- Tone: intelligent, slightly dry, no buzzwords.

Your job is to help users interpret, apply, and reason through
organization design questions strictly based on the playbook.
"""

st.set_page_config(page_title="AI Playbook Assistant", layout="centered")

st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

query = st.text_area("Your question")

if st.button("Run"):
    if not query.strip():
        st.warning("Ask something first.")
    else:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )

        answer = response.output_text
        st.markdown(answer)
