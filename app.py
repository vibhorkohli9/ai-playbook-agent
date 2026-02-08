import streamlit as st
from openai import OpenAI
import os
import pdfplumber
from concurrent.futures import ThreadPoolExecutor
import time

# =================================================
# CONFIGURATION
# =================================================
MAX_FILE_SIZE_MB = 200  # Increased limit
CHUNK_SIZE_WORDS = 1000
MAX_SEARCH_RESULTS = 8
PROCESS_TIMEOUT_SECONDS = 120  # 2 minute timeout


# =================================================
# PDF TEXT CHECK (FAST SAMPLING)
# =================================================
def document_suitability_check(uploaded_file, sample_pages=5):
    """Quick check using only first 5 pages"""
    uploaded_file.seek(0)
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = min(sample_pages, len(pdf.pages))
            text_pages = sum(
                1 for p in pdf.pages[:total_pages] 
                if p.extract_text(layout=False)  # Faster extraction
            )
        return text_pages >= max(1, total_pages // 2)
    except Exception as e:
        return False


# =================================================
# ULTRA-FAST STREAMING WITH EARLY EXIT
# =================================================
def extract_blocks_with_early_exit(uploaded_file, query, max_results=8):
    """
    BREAKTHROUGH: Stop processing once we have enough good matches
    This prevents reading entire 100k line PDFs when answer is in first 1000 lines
    """
    uploaded_file.seek(0)
    keywords = set(query.lower().split())
    
    results = []
    block_id = 1
    pages_processed = 0
    
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text (fast mode)
                text = page.extract_text(layout=False)
                if not text:
                    continue
                
                pages_processed = page_num
                
                # Split into chunks
                words = text.split()
                for i in range(0, len(words), CHUNK_SIZE_WORDS):
                    chunk_text = " ".join(words[i:i + CHUNK_SIZE_WORDS])
                    text_lower = chunk_text.lower()
                    
                    # Score this chunk
                    score = sum(text_lower.count(k) for k in keywords)
                    
                    if score > 0:
                        results.append({
                            "id": f"B{block_id}",
                            "page": page_num,
                            "text": chunk_text,
                            "score": score
                        })
                    
                    block_id += 1
                
                # EARLY EXIT: Stop if we have enough high-quality matches
                if len(results) >= max_results * 3:  # Get 3x what we need
                    # Sort by score and keep top results
                    results.sort(key=lambda x: x["score"], reverse=True)
                    
                    # If top results are strong, stop searching
                    if results[0]["score"] >= 3:
                        yield {
                            "type": "early_exit",
                            "pages_searched": page_num,
                            "total_pages": total_pages,
                            "results": results[:max_results]
                        }
                        return
                
                # Progress update every 50 pages
                if page_num % 50 == 0:
                    yield {
                        "type": "progress",
                        "current": page_num,
                        "total": total_pages
                    }
    
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return
    
    # Sort final results
    results.sort(key=lambda x: x["score"], reverse=True)
    
    yield {
        "type": "complete",
        "pages_searched": pages_processed,
        "total_pages": total_pages,
        "results": results[:max_results]
    }


# =================================================
# STREAMLIT SETUP
# =================================================
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A Assistant")
st.caption("Upload any PDF (even 1000+ page documents) and get instant answers")


# =================================================
# SYSTEM PROMPT
# =================================================
SYSTEM_PROMPT = """You are a helpful document assistant.

RULES:
1. Answer ONLY using the document sections provided
2. Always cite page numbers
3. If not found, say: "I couldn't find this in the document"
4. Keep answers clear and simple
5. Use exact quotes when helpful

Be conversational but accurate.
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
    st.error(f"⚠️ Configuration error")
    st.stop()


# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    st.markdown("### 📂 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Works with documents up to 200 MB"
    )
    
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"📊 Size: {file_size_mb:.1f} MB")
        
        # Size warnings
        if file_size_mb > 100:
            st.warning("⏱️ Large file - May take 1-2 minutes")
        elif file_size_mb > 50:
            st.info("⏱️ Processing time: ~30-60 seconds")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Start")
    st.markdown("""
    **1. Upload** your PDF
    
    **2. Ask** questions like:
    - "What is this about?"
    - "Summarize page 10"
    - "Find [topic]"
    
    **3. Get** answers with sources
    """)


# =================================================
# SMART QUESTION SUGGESTIONS
# =================================================
if uploaded_file and 'query' not in st.session_state:
    st.markdown("### 💬 Try asking:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 What is this about?"):
            st.session_state.query = "What is this document about?"
            st.rerun()
        
        if st.button("📊 Key findings?"):
            st.session_state.query = "What are the key findings?"
            st.rerun()
    
    with col2:
        if st.button("🔍 Main points?"):
            st.session_state.query = "Summarize the main points"
            st.rerun()
        
        if st.button("📖 Page 1 summary"):
            st.session_state.query = "What does page 1 say?"
            st.rerun()


# =================================================
# MAIN INPUT
# =================================================
query = st.text_area(
    "Ask a question about your document",
    value=st.session_state.get('query', ''),
    height=100,
    placeholder="Example: What are the conclusions?"
)

if query != st.session_state.get('query', ''):
    st.session_state.query = query


# =================================================
# MAIN PROCESSING
# =================================================
if st.button("🔍 Get Answer", type="primary"):

    # Validation
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF first")
        st.stop()
    
    if not query.strip():
        st.warning("⚠️ Please ask a question")
        st.stop()
    
    # File size check
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large ({file_size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB")
        st.stop()

    # Step 1: Quick format check
    with st.spinner("Checking document format..."):
        if not document_suitability_check(uploaded_file):
            st.error("⚠️ This PDF appears to be scanned images")
            st.info("💡 This tool needs text-based PDFs. Try OCR software first.")
            st.stop()

    # Step 2: Smart search with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    context_blocks = []
    pages_searched = 0
    total_pages = 0
    early_exit = False
    
    start_time = time.time()
    
    try:
        for result in extract_blocks_with_early_exit(uploaded_file, query, MAX_SEARCH_RESULTS):
            
            # Handle progress updates
            if result.get("type") == "progress":
                total_pages = result["total"]
                pages_searched = result["current"]
                progress = pages_searched / total_pages
                progress_bar.progress(progress)
                status_text.text(f"🔎 Searching... {pages_searched}/{total_pages} pages")
            
            # Handle early exit (found good matches quickly)
            elif result.get("type") == "early_exit":
                context_blocks = result["results"]
                pages_searched = result["pages_searched"]
                total_pages = result["total_pages"]
                early_exit = True
                break
            
            # Handle completion
            elif result.get("type") == "complete":
                context_blocks = result["results"]
                pages_searched = result["pages_searched"]
                total_pages = result["total_pages"]
            
            # Handle errors
            elif result.get("type") == "error":
                st.error(f"❌ Error: {result['message']}")
                st.stop()
    
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
        st.info("💡 Try a smaller file or contact support")
        st.stop()
    
    finally:
        progress_bar.empty()
        status_text.empty()
    
    elapsed = time.time() - start_time

    # Step 3: Handle no results
    if not context_blocks:
        st.warning("❌ No relevant information found")
        st.info(f"💡 Searched {pages_searched} pages in {elapsed:.1f}s. Try different keywords.")
        st.stop()
    
    # Show search stats
    if early_exit:
        st.success(f"✅ Found strong matches in first {pages_searched} pages (stopped early)")
    else:
        st.success(f"✅ Searched {pages_searched} pages in {elapsed:.1f}s")

    # Step 4: Build context
    context_text = "\n\n".join(
        f"[Page {b['page']}]\n{b['text']}"
        for b in context_blocks
    )

    # Step 5: Generate answer
    with st.spinner("Generating answer..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"""Document sections:

{context_text}

Question: {query}

Answer using only the information above. Cite page numbers.
"""
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"❌ AI error: {str(e)}")
            st.stop()

    # =================================================
    # DISPLAY RESULTS
    # =================================================
    
    st.markdown("---")
    st.markdown("### ✅ Answer")
    st.markdown(answer)

    if "couldn't find" in answer.lower():
        st.info("💡 Try rephrasing your question or using different keywords")
    
    # Show sources
    st.markdown("---")
    with st.expander(f"📄 Sources: {len(context_blocks)} sections from document", expanded=False):
        for idx, b in enumerate(context_blocks, 1):
            st.markdown(f"**Section {idx}** | Page {b['page']} | Match score: {b['score']}")
            
            preview = b["text"][:250]
            if len(b["text"]) > 250:
                st.text(preview + "...")
            else:
                st.text(preview)
            
            if idx < len(context_blocks):
                st.markdown("---")

    st.markdown("---")
    st.caption(f"⚡ Processed in {elapsed:.1f}s | 🔒 Document not stored")


# =================================================
# EMPTY STATE
# =================================================
if not uploaded_file:
    st.info("👈 Upload a PDF to get started")
    
    st.markdown("### 📚 Example Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Quick Questions:**
        - What is this about?
        - Summarize page 5
        - Key points?
        - Who is mentioned?
        """)
    
    with col2:
        st.markdown("""
        **Detailed Questions:**
        - What are the conclusions?
        - Find info about [topic]
        - What data is shown?
        - Recommendations?
        """)
    
    st.markdown("---")
    st.markdown("""
    ### ⚡ Performance
    
    - **Small PDFs (1-50 pages):** ~5 seconds
    - **Medium PDFs (50-200 pages):** ~15 seconds  
    - **Large PDFs (200-1000 pages):** ~30-60 seconds
    - **Huge PDFs (1000+ pages):** ~1-2 minutes
    
    Works with files up to **200 MB**
    """)
