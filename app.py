import streamlit as st
from openai import OpenAI
import os
import pdfplumber

# =================================================
# PDF TEXT CHECK
# =================================================
def document_suitability_check(uploaded_file, sample_pages=10):
    """Check if PDF contains extractable text"""
    uploaded_file.seek(0)
    with pdfplumber.open(uploaded_file) as pdf:
        text_pages = sum(
            1 for p in pdf.pages[:sample_pages] if p.extract_text()
        )
    return text_pages >= max(1, sample_pages // 3)


# =================================================
# STREAMING BLOCK EXTRACTION (MEMORY SAFE)
# =================================================
def extract_text_blocks_streaming(uploaded_file, block_size=1000):
    """
    Generator that yields blocks on-demand instead of loading all into memory.
    This prevents memory crashes with large PDFs.
    """
    uploaded_file.seek(0)
    block_id = 1

    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)
        
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            words = text.split()
            for i in range(0, len(words), block_size):
                yield {
                    "id": f"B{block_id}",
                    "page": page_num,
                    "text": " ".join(words[i:i + block_size])
                }
                block_id += 1
            
            # Yield progress info every 100 pages
            if page_num % 100 == 0:
                yield {"type": "progress", "current": page_num, "total": total_pages}


# =================================================
# TWO-PASS SMART FILTERING (PERFORMANCE FIX)
# =================================================
def smart_filter_blocks(uploaded_file, query, progress_placeholder):
    """
    PASS 1: Quick keyword filter (eliminates 90%+ of irrelevant blocks)
    PASS 2: Detailed scoring only on candidates
    """
    keywords = set(query.lower().split())
    
    # PASS 1: Fast filter
    candidate_blocks = []
    
    for item in extract_text_blocks_streaming(uploaded_file):
        # Handle progress updates
        if item.get("type") == "progress":
            progress_placeholder.text(f"Reading pages {item['current']}/{item['total']}...")
            continue
        
        text_lower = item["text"].lower()
        
        # Quick check: does this block contain ANY keyword?
        if any(k in text_lower for k in keywords):
            candidate_blocks.append(item)
    
    progress_placeholder.empty()
    
    # If no matches found
    if not candidate_blocks:
        return []
    
    # PASS 2: Score candidates
    scored = []
    for block in candidate_blocks:
        text_lower = block["text"].lower()
        score = sum(text_lower.count(k) for k in keywords)
        scored.append((score, block))
    
    # Sort by score and return top 10
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in scored[:10]]


# =================================================
# STREAMLIT SETUP
# =================================================
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A Assistant")
st.caption("Upload any PDF and ask questions. Get answers instantly with sources.")


# =================================================
# SYSTEM PROMPT
# =================================================
SYSTEM_PROMPT = """You are a helpful document assistant.

RULES:
1. Answer ONLY using the document sections provided below
2. Always mention which page or section you found the information
3. If the answer is not in the document, say: "I couldn't find this information in the document."
4. Keep answers clear and simple
5. Use exact quotes when possible

Be helpful and conversational while staying accurate to the document.
"""


# =================================================
# OPENAI CLIENT
# =================================================
try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
except Exception as e:
    st.error(f"⚠️ Configuration error. Please contact support.")
    st.stop()


# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    st.markdown("### 📂 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a text-based PDF (not scanned images)"
    )
    
    if uploaded_file:
        # Show file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.caption(f"Size: {file_size_mb:.1f} MB")
        
        # File size warning
        if file_size_mb > 50:
            st.warning("⚠️ Large file detected. Processing may take 30-60 seconds.")
    
    st.markdown("---")
    st.markdown("### 💡 How to use")
    st.markdown("""
    **1. Upload** your PDF document
    
    **2. Ask** a question like:
    - "What is this document about?"
    - "Summarize page 5"
    - "Find information about [topic]"
    - "What are the main points?"
    
    **3. Get** instant answers with page references
    """)


# =================================================
# SUGGESTION PROMPTS (AFTER UPLOAD)
# =================================================
if uploaded_file and 'query' not in st.session_state:
    st.markdown("### 💬 Try asking:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 What is this document about?"):
            st.session_state.query = "What is this document about?"
            st.rerun()
        
        if st.button("🔍 Summarize the main points"):
            st.session_state.query = "Summarize the main points"
            st.rerun()
    
    with col2:
        if st.button("📊 What are the key findings?"):
            st.session_state.query = "What are the key findings?"
            st.rerun()
        
        if st.button("❓ What does page 1 say?"):
            st.session_state.query = "What does page 1 say?"
            st.rerun()


# =================================================
# MAIN INPUT AREA
# =================================================
query = st.text_area(
    "Ask a question about your document",
    value=st.session_state.get('query', ''),
    height=100,
    placeholder="Example: What are the conclusions on page 10?"
)

# Clear session state if query changes
if query != st.session_state.get('query', ''):
    st.session_state.query = query


# =================================================
# MAIN PROCESSING BUTTON
# =================================================
if st.button("🔍 Get Answer", type="primary"):

    # Validation
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF document first.")
        st.stop()
    
    if not query.strip():
        st.warning("⚠️ Please type a question.")
        st.stop()

    # Step 1: Check document format
    with st.spinner("Checking your document..."):
        if not document_suitability_check(uploaded_file):
            st.error("⚠️ This PDF appears to be a scanned image.")
            st.info("💡 This tool works with text-based PDFs. Please try a different document.")
            st.stop()

    # Step 2: Search for relevant sections
    progress_placeholder = st.empty()
    progress_placeholder.text("🔎 Searching through your document...")
    
    try:
        context_blocks = smart_filter_blocks(uploaded_file, query, progress_placeholder)
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        st.info("💡 Try a smaller PDF or contact support if this continues.")
        st.stop()

    # Step 3: Handle no results
    if not context_blocks:
        st.warning("❌ I couldn't find relevant information for your question.")
        st.info("💡 **Try:**\n- Using different words\n- Making your question more specific\n- Checking if that topic is in the document")
        st.stop()

    # Step 4: Build context for AI
    context_text = "\n\n".join(
        f"[Page {b['page']}]\n{b['text']}"
        for b in context_blocks
    )

    # Step 5: Generate answer
    with st.spinner("Writing your answer..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"""Here are relevant sections from the document:

{context_text}

User's question: {query}

Please answer the question using only the information above. Mention page numbers when possible.
"""
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            st.stop()

    # =================================================
    # DISPLAY RESULTS
    # =================================================
    
    st.markdown("---")
    st.markdown("### ✅ Answer")
    st.markdown(answer)

    # Handle "not found" response
    if "couldn't find" in answer.lower():
        st.info("💡 The information you're looking for might not be in this document. Try rephrasing your question.")
        st.stop()

    # Show where answer came from
    st.markdown("---")
    with st.expander(f"📄 Source: Found in {len(context_blocks)} sections", expanded=False):
        st.caption("These are the parts of your document that were used to answer your question")
        
        for idx, b in enumerate(context_blocks, 1):
            st.markdown(f"**Section {idx}** | Page {b['page']}")
            
            # Show preview
            preview_text = b["text"][:300]
            if len(b["text"]) > 300:
                st.text(preview_text + "...")
            else:
                st.text(preview_text)
            
            if idx < len(context_blocks):
                st.markdown("---")

    # Footer
    st.markdown("---")
    st.caption("✅ Answer generated from your document | 🔒 Your document is not stored")


# =================================================
# EMPTY STATE
# =================================================
if not uploaded_file:
    st.info("👈 Upload a PDF from the sidebar to get started")
    
    st.markdown("### 📚 Example Questions You Can Ask:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Simple Questions:**
        - What is this about?
        - Summarize page 3
        - What are the key points?
        - Who is mentioned in this document?
        """)
    
    with col2:
        st.markdown("""
        **Detailed Questions:**
        - What are the main conclusions?
        - Find information about [topic]
        - What recommendations are given?
        - What data is shown in the charts?
        """)
    
    st.markdown("---")
    st.markdown("### 📋 Works best with:")
    st.markdown("""
    - Reports and research papers
    - Contracts and legal documents
    - Manuals and guides
    - Books and articles
    - Any text-based PDF up to **100 MB**
    """)
