import streamlit as st
import google.generativeai as genai
import os

# --- SYSTEM PROMPT ---
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

Your job is to help users interpret, apply, and reason through
organization design questions strictly based on the playbook.
"""

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Playbook Assistant", layout="centered")

st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

question = st.text_area("Your question")

# --- GEMINI SETUP ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

# --- RUN BUTTON ---
if st.button("Run"):
    if not question.strip():
        st.warning("Ask something first.")
    else:
        with st.spinner("Thinking…"):
            response = model.generate_content(question)
            st.markdown(response.text)
