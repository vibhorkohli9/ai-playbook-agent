import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="AI Playbook Assistant", layout="centered")

st.title("AI Playbook Assistant")
st.caption("Ask a question. Get a grounded answer. No hallucinations.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

query = st.text_area("Your question")

if st.button("Run"):
    if not query.strip():
        st.warning("Ask something first.")
    else:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional, accurate AI assistant."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )

        st.markdown(response.choices[0].message.content)
