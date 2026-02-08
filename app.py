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
MAX_CONTEXT_BLOCKS = 15  # Use more blocks for better accuracy


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
        --white: #FFFFFF;
        --light-gray: #F3F4F6;
    }
    
    /* Header styling */
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
    
    /* Progress bar container */
    .progress-container {
        background: white;
        border: 2px solid #DC2626;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.1);
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
    
    .confidence-high {
        background: #10B981;
        color: white;
    }
    
    .confidence-medium {
        background: #F59E0B;
        color: white;
    }
    
    .confidence-low {
        background: #EF4444;
        color: white;
    }
    
    /* Answer box */
    .answer-box {
        background: white;
        border-left: 4px solid #DC2626;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Source section */
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
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #DC2626 !important;
        border-radius: 8px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #F9FAFB;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
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
# CONFIDENCE SCORE CALCULATOR
# =================================================
def calculate_confidence(blocks, query):
    """Calculate user-friendly confidence score"""
    if not blocks:
        return 0, "Low"
    
    keywords = set(query.lower().split())
    
    # Count total keyword occurrences across all blocks
    total_matches = 0
    for block in blocks:
        text_lower = block["text"].lower()
        total_matches += sum(text_lower.count(k) for k in keywords)
    
    # Calculate confidence percentage
    # More matches = higher confidence
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
            # Check first 3 pages
            for page in pdf.pages[:3]:
                text = page.extract_text(layout=False)
                if text and len(text.strip()) > 50:
                    return True
        return False
    except:
        return False


# =================================================
# FULL DOCUMENT SEARCH (NO EARLY EXIT)
# =================================================
def search_entire_document(uploaded_file, query, progress_placeholder):
    """
    CRITICAL: Search the ENTIRE document, no shortcuts
    This ensures we don't miss answers on page 218 or anywhere else
    """
    uploaded_file.seek(0)
    keywords = set(query.lower().split())
    
    all_blocks = []
    block_id = 1
    
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text(layout=False)
                if not text:
                    continue
                
                # Split into chunks
                words = text.split()
                for i in range(0, len(words), CHUNK_SIZE_WORDS):
                    chunk_text = " ".join(words[i:i + CHUNK_SIZE_WORDS])
                    text_lower = chunk_text.lower()
                    
                    # Score this chunk
                    score = sum(text_lower.count(k) for k in keywords)
                    
                    # Keep ALL blocks with any matches
                    if score > 0:
                        all_blocks.append({
                            "id": f"B{block_id}",
                            "page": page_num,
                            "text": chunk_text,
                            "score": score
                        })
                    
                    block_id += 1
                
                # Update progress every 20 pages
                if page_num % 20 == 0 or page_num == total_pages:
                    progress_html = show_animated_progress(
                        page_num, 
                        total_pages, 
                        "Scanning document..."
                    )
                    progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
        
        # Sort by score and return top results
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

# Inject custom styling
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
        
        if file_size_mb > 100:
            st.warning("⏱️ Large file - Processing will take 1-2 minutes")
        elif file_size_mb > 50:
            st.info("⏱️ Processing time: ~30-60 seconds")
    
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown("""
    **1.** Upload your PDF document
    
    **2.** Ask any question
    
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
# SEARCH BUTTON
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

    # Format check
    with st.spinner("Checking document..."):
        if not check_pdf_format(uploaded_file):
            st.error("⚠️ This PDF appears to be scanned images, not text")
            st.info("💡 Please use a text-based PDF or convert with OCR software")
            st.stop()

    # Full document search
    progress_placeholder = st.empty()
    
    start_time = time.time()
    
    try:
        context_blocks, total_pages = search_entire_document(
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
    
    # Show search completion
    st.success(f"✅ Searched entire document ({total_pages} pages) in {elapsed:.1f}s")

    # Build context for AI
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
    # DISPLAY ANSWER WITH CONFIDENCE
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

    # Warning if low confidence
    if confidence_level == "Low":
        st.info("💡 Low confidence - The answer may not be fully accurate. Try rephrasing your question.")
    
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
    st.caption(f"⚡ Processed in {elapsed:.1f}s • 🔒 Your document is not stored • 📄 Searched {total_pages} pages")


# =================================================
# EMPTY STATE
# =================================================
if not uploaded_file:
    st.info("👈 Upload a PDF from the sidebar to get started")
    
    st.markdown("### 📚 Example Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Quick Questions:**
        - What is this about?
        - Summarize page 10
        - Key findings?
        - Main conclusions?
        """)
    
    with col2:
        st.markdown("""
        **Specific Questions:**
        - What does page 218 say about [topic]?
        - Find information about [specific term]
        - What are the recommendations?
        - What data is presented?
        """)
    
    st.markdown("---")
    st.markdown("### ⚡ How It Works")
    st.markdown("""
    1. **Upload** your PDF (up to 200 MB)
    2. **We scan** every single page - nothing is skipped
    3. **Get answers** with confidence scores and page numbers
    
    **Supported:**
    - Small PDFs (1-50 pages): ~5 seconds
    - Medium PDFs (50-200 pages): ~15-30 seconds
    - Large PDFs (200-1000+ pages): ~1-2 minutes
    
    **We search your entire document** - If the answer is on page 218, we'll find it!
    """)
