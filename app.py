import streamlit as st
import google.generativeai as genai
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Playbook Assistant", layout="centered")

st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

# ---------- SYSTEM PROMPT ----------
SYSTEM_PROMPT = """
You are an Organization Design Consultant AI.

Rules you MUST follow:
- Answer ONLY using the provided playbook content.
- If the answer is not explicitly present, say:
  "This is not covered in the playbook."
- Do NOT use external knowledge.
- Do NOT hallucinate.
- Be clear, grounded, and practical.
- Tone: intelligent, slightly dry, no buzzwords.
"""

# ---------- GEMINI SETUP ----------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="models/gemini-pro",
    system_instruction=SYSTEM_PROMPT
)

# ---------- UI ----------
query = st.text_area("Your question")

if st.button("Run"):
    if not query.strip():
        st.warning("Ask something first.")
    else:
        with st.spinner("Thinking (responsibly)..."):
            response = model.generate_content(query)

        st.markdown(response.text)
