import streamlit as st
from openai import OpenAI
import os
import pdfplumber
import time

# =================================================
# CONFIGURATION
# =================================================
MAX_FILE_SIZE_MB = 200
CHUNK_SIZE_WORDS = 800
MAX_CONTEXT_BLOCKS = 15
STREAMLIT_TIMEOUT_SECONDS = 85  # Safe limit (Streamlit Cloud = 90s)
PROCESSING_SPEED_PAGES_PER_SECOND = 5  # Conservative estimate


# =================================================
# CUSTOM CSS - RED & WHITE THEME
# =================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-red: #DC2626;
        --light-red: #FEE2E2;
        --dark-red: #991B1B;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #FEE2E2;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Progress container */
    .progress-container {
        background: white;
        border: 2px solid #DC2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.1);
    }
    
    /* Time estimate box */
    .time-estimate {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 4px solid #DC2626;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .confidence-high { background: #10B981; color: white; }
    .confidence-medium { background: #F59E0B; color: white; }
    .confidence-low { background: #EF4444; color: white; }
    
    /* Answer box */
    .answer-box {
        background: white;
        border-left: 4px solid #DC2626;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Source item */
    .source-item {
        background: #F9FAFB;
        border-left: 3px solid #DC2626;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(220, 38, 38, 0.2);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(220, 38, 38, 0.3);
    }
    
    /* Text input */
    .stTextArea>div>div>textarea {
        border: 2px solid #E5E7EB;
        border-radius: 8px;
        font-size: 1rem;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #DC2626;
        box-shadow: 0 0 0 1px #DC2626;
    }
    </style>
    """, unsafe_allow_html=True)


# =================================================
# ESTIMATE PROCESSING TIME
# =================================================
def estimate_processing_time(uploaded_file):
    """
    Calculate estimated time BEFORE processing
    Returns: (estimated_seconds, total_pages, should_proceed)
    """
    uploaded_file.seek(0)
    
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = len(pdf.pages)
        
        # Calculate estimated time
        # Formula: pages / processing_speed + overhead
        estimated_time = (total_pages / PROCESSING_SPEED_PAGES_PER_SECOND) + 5  # +5s overhead
        
        # Check if within timeout limit
        should_proceed = estimated_time <= STREAMLIT_TIMEOUT_SECONDS
        
        return estimated_time, total_pages, should_proceed
        
    except Exception as e:
        return None, None, False


# =================================================
# ESTIMATE BY LINES (ALTERNATIVE)
# =================================================
def estimate_by_file_size(file_size_mb):
    """
    Quick estimation based on file size
    Average: 1 MB ≈ 10 pages ≈ 2800 lines
    """
    estimated_pages = int(file_size_mb * 10)
    estimated_lines = int(file_size_mb * 2800)
    estimated_time = (estimated_pages / PROCESSING_SPEED_PAGES_PER_SECOND) + 5
    
    should_proceed = estimated_time <= STREAMLIT_TIMEOUT_SECONDS
    
    return estimated_time, estimated_pages, estimated_lines, should_proceed


# =================================================
# ANIMATED PROGRESS BAR
# =================================================
def show_animated_progress(current, total, status_text):
    """Beautiful animated progress with wave effect"""
    progress_html = f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; color: #374151;">📖 {status_text}</span>
            <span style="color: #6B7280;">{current}/{total} pages</span>
        </div>
        <div style="background: #E5E7EB; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="
                background: linear-gradient(90deg, #DC2626 0%, #EF4444 50%, #DC2626 100%);
                height: 100%;
                width: {(current/total)*100}%;
                border-radius: 4px;
                animation: wave 1.5s ease-in-out infinite;
                background-size: 200% 100%;
            "></div>
        </div>
    </div>
    
    <style>
    @keyframes wave {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    </style>
    """
    return progress_html


# =================================================
# CONFIDENCE CALCULATOR
# =================================================
def calculate_confidence(blocks, query):
    """Calculate user-friendly confidence score"""
    if not blocks:
        return 0, "Low"
    
    keywords = set(query.lower().split())
    total_matches = sum(
        sum(block["text"].lower().count(k) for k in keywords)
        for block in blocks
    )
    
    if total_matches >= 20:
        confidence = min(95, 70 + (total_matches - 20))
        level = "High"
    elif total_matches >= 10:
        confidence = 50 + (total_matches - 10) * 2
        level = "Medium"
    else:
        confidence = total_matches * 5
        level = "Low"
    
    return min(confidence, 99), level


# =================================================
# PDF FORMAT CHECK
# =================================================
def check_pdf_format(uploaded_file):
    """Quick check if PDF has text"""
    uploaded_file.seek(0)
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages[:3]:
                text = page.extract_text(layout=False)
                if text and len(text.strip()) > 50:
                    return True
        return False
    except:
        return False


# =================================================
# OPTIMIZED FULL DOCUMENT SEARCH
# =================================================
def search_entire_document_optimized(uploaded_file, query, progress_placeholder):
    """
    Optimized full document search with:
    - Batch processing
    - Faster extraction
    - Page-level keyword filtering
    """
    uploaded_file.seek(0)
    keywords = set(query.lower().split())
    
    all_blocks = []
    block_id = 1
    
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = len(pdf.pages)
            batch_size = 25
            
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                batch_pages = pdf.pages[batch_start:batch_end]
                
                for page_num, page in enumerate(batch_pages, start=batch_start + 1):
                    # Fast extraction
                    text = page.extract_text(layout=False, x_tolerance=3, y_tolerance=3)
                    if not text or len(text) < 50:
                        continue
                    
                    text_lower = text.lower()
                    
                    # Skip pages without keywords
                    has_keywords = any(k in text_lower for k in keywords)
                    if not has_keywords:
                        continue
                    
                    # Chunk only relevant pages
                    words = text.split()
                    for i in range(0, len(words), CHUNK_SIZE_WORDS):
                        chunk_text = " ".join(words[i:i + CHUNK_SIZE_WORDS])
                        chunk_lower = chunk_text.lower()
                        
                        score = sum(chunk_lower.count(k) for k in keywords)
                        
                        if score > 0:
                            all_blocks.append({
                                "id": f"B{block_id}",
                                "page": page_num,
                                "text": chunk_text,
                                "score": score
                            })
                        
                        block_id += 1
                
                # Update progress
                progress_html = show_animated_progress(
                    batch_end, 
                    total_pages, 
                    "Scanning document..."
                )
                progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
        
        all_blocks.sort(key=lambda x: x["score"], reverse=True)
        return all_blocks[:MAX_CONTEXT_BLOCKS], total_pages
        
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


# =================================================
# STREAMLIT APP
# =================================================
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="centered"
)

inject_custom_css()

# Header
st.markdown("""
<div class="main-header">
    <h1>📄 Document Assistant</h1>
    <p>Upload any PDF and get accurate answers from your entire document</p>
</div>
""", unsafe_allow_html=True)


# =================================================
# OPENAI CLIENT
# =================================================
try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
except:
    st.error("⚠️ Configuration error. Please contact support.")
    st.stop()


# =================================================
# SIDEBAR
# =================================================
with st.sidebar:
    st.markdown("### 📂 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Maximum size: 200 MB"
    )
    
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"📊 {file_size_mb:.1f} MB")
        
        # Quick estimate by file size
        est_time, est_pages, est_lines, can_process = estimate_by_file_size(file_size_mb)
        
        # Show estimate
        if can_process:
            st.markdown(f"""
            <div class="time-estimate">
                <strong>⏱️ Estimated Processing Time</strong><br>
                📄 ~{est_pages} pages (~{est_lines:,} lines)<br>
                ⏰ ~{int(est_time)} seconds
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Document Too Large</strong><br>
                📄 Estimated: ~{est_pages} pages<br>
                ⏰ Would take: ~{int(est_time)} seconds<br>
                🚫 Limit: {STREAMLIT_TIMEOUT_SECONDS} seconds<br><br>
                <strong>Please use a smaller PDF (under ~400 pages)</strong>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown("""
    **1.** Upload your PDF document
    
    **2.** Ask any question related to the document only
    
    **3.** Get accurate answers with page numbers
    
    We search your **entire document** to find the best answer.
    """)


# =================================================
# SUGGESTION BUTTONS
# =================================================
if uploaded_file and 'query' not in st.session_state:
    st.markdown("### 💬 Quick Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 What's this about?", use_container_width=True):
            st.session_state.query = "What is this document about?"
            st.rerun()
        
        if st.button("🔍 Main points", use_container_width=True):
            st.session_state.query = "What are the main points?"
            st.rerun()
    
    with col2:
        if st.button("📊 Key findings", use_container_width=True):
            st.session_state.query = "What are the key findings?"
            st.rerun()
        
        if st.button("📖 Summary", use_container_width=True):
            st.session_state.query = "Give me a summary"
            st.rerun()


# =================================================
# MAIN INPUT
# =================================================
query = st.text_area(
    "Ask a question",
    value=st.session_state.get('query', ''),
    height=100,
    placeholder="Example: What does the document say about financial results?",
    label_visibility="collapsed"
)

if query != st.session_state.get('query', ''):
    st.session_state.query = query


# =================================================
# SEARCH BUTTON WITH PRE-CHECK
# =================================================
if st.button("🔍 Search Document", type="primary", use_container_width=True):

    # Validation
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF document first")
        st.stop()
    
    if not query.strip():
        st.warning("⚠️ Please ask a question")
        st.stop()
    
    # File size check
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ File too large. Maximum: {MAX_FILE_SIZE_MB} MB")
        st.stop()

    # ============================================
    # CRITICAL: CHECK TIME BEFORE PROCESSING
    # ============================================
    with st.spinner("Analyzing document size..."):
        estimated_time, total_pages, should_proceed = estimate_processing_time(uploaded_file)
    
    if not should_proceed:
        st.markdown(f"""
        <div class="warning-box">
            <h3 style="margin-top: 0; color: #DC2626;">⚠️ Document Too Large to Process</h3>
            <p><strong>Your document:</strong></p>
            <ul>
                <li>📄 Pages: {total_pages}</li>
                <li>⏰ Estimated time: ~{int(estimated_time)} seconds</li>
                <li>🚫 Processing limit: {STREAMLIT_TIMEOUT_SECONDS} seconds</li>
            </ul>
            <p><strong>Solution:</strong></p>
            <ul>
                <li>✅ Use a PDF with fewer than 400 pages</li>
                <li>✅ Split your PDF into smaller parts</li>
                <li>✅ Ask about specific page ranges (e.g., "What does page 50-100 say?")</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Show time estimate
    st.markdown(f"""
    <div class="time-estimate">
        <strong>⏱️ Processing Information</strong><br>
        📄 Total pages: {total_pages}<br>
        ⏰ Estimated time: ~{int(estimated_time)} seconds<br>
        ✅ Within processing limit
    </div>
    """, unsafe_allow_html=True)

    # Format check
    with st.spinner("Checking document format..."):
        if not check_pdf_format(uploaded_file):
            st.error("⚠️ This PDF appears to be scanned images, not text")
            st.info("💡 Please use a text-based PDF or convert with OCR software")
            st.stop()

    # Full document search
    progress_placeholder = st.empty()
    start_time = time.time()
    
    try:
        context_blocks, total_pages = search_entire_document_optimized(
            uploaded_file, 
            query, 
            progress_placeholder
        )
    except Exception as e:
        progress_placeholder.empty()
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    progress_placeholder.empty()
    elapsed = time.time() - start_time

    # No results
    if not context_blocks:
        st.warning("❌ No information found in the document")
        st.info(f"💡 Searched all {total_pages} pages in {elapsed:.1f}s. Try different keywords.")
        st.stop()
    
    # Calculate confidence
    confidence_score, confidence_level = calculate_confidence(context_blocks, query)
    
    # Show completion
    st.success(f"✅ Searched entire document ({total_pages} pages) in {elapsed:.1f}s")

    # Build context
    context_text = "\n\n".join(
        f"[Page {b['page']}]\n{b['text']}"
        for b in context_blocks
    )

    # Generate answer
    with st.spinner("Analyzing and writing answer..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a document assistant. 

RULES:
1. Answer using ONLY the information provided
2. Always cite page numbers
3. Be accurate and specific
4. If not found, say "This information is not in the document"
5. Use direct quotes when helpful"""
                    },
                    {
                        "role": "user",
                        "content": f"""Document sections:

{context_text}

Question: {query}

Provide a clear answer with page citations.
"""
                    }
                ],
                temperature=0.2,
                max_tokens=1200
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            st.stop()

    # =================================================
    # DISPLAY ANSWER
    # =================================================
    
    st.markdown("---")
    
    # Confidence badge
    if confidence_level == "High":
        badge_class = "confidence-high"
        emoji = "✅"
    elif confidence_level == "Medium":
        badge_class = "confidence-medium"
        emoji = "⚠️"
    else:
        badge_class = "confidence-low"
        emoji = "❌"
    
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="confidence-badge {badge_class}">
            {emoji} Confidence: {confidence_score}% ({confidence_level})
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    if confidence_level == "Low":
        st.info("💡 Low confidence - Try rephrasing your question.")
    
    # Show sources
    st.markdown("---")
    with st.expander(f"📄 View {len(context_blocks)} Source Sections", expanded=False):
        st.caption("These sections from your document were used to create the answer")
        
        for idx, block in enumerate(context_blocks, 1):
            st.markdown(f"""
            <div class="source-item">
                <strong>Section {idx}</strong> • Page {block['page']}
            </div>
            """, unsafe_allow_html=True)
            
            preview = block["text"][:300]
            if len(block["text"]) > 300:
                st.text(preview + "...")
            else:
                st.text(preview)
            
            if idx < len(context_blocks):
                st.markdown("")

    # Footer
    st.markdown("---")
    st.caption(f"⚡ Processed in {elapsed:.1f}s • 🔒 Document not stored • 📄 Searched {total_pages} pages")


# =================================================
# EMPTY STATE (NO "Example Questions" SECTION)
# =================================================
if not uploaded_file:
    st.info("👈 Upload a PDF from the sidebar to get started")
    
    st.markdown("---")
    st.markdown("### ⚡ How It Works")
    st.markdown("""
    1. **Upload** your PDF (up to 200 MB)
    2. **We scan** every single page - nothing is skipped
    3. **Get answers** with confidence scores and page numbers
    
    **Processing Speed:**
    - Small PDFs (1-50 pages): ~5-10 seconds
    - Medium PDFs (50-200 pages): ~15-30 seconds
    - Large PDFs (200-400 pages): ~40-80 seconds
    
    **Maximum:** ~400 pages due to 90-second processing limit
    
    **We search your entire document** - If the answer is on page 218, we'll find it!
    """)
