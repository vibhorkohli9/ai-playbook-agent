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
def extract_text_blocks_streaming(uploaded_file, block_size=800):
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
            
            # Yield progress info every 50 pages
            if page_num % 50 == 0:
                yield {"type": "progress", "current": page_num, "total": total_pages}


# =================================================
# TWO-PASS SMART FILTERING (PERFORMANCE FIX)
# =================================================
def smart_filter_blocks(uploaded_file, query, max_blocks=12):
    """
    PASS 1: Quick keyword filter (eliminates 90%+ of irrelevant blocks)
    PASS 2: Detailed scoring only on candidates
    
    This prevents scoring all 50,000+ blocks in a large PDF.
    """
    keywords = set(query.lower().split())
    
    # PASS 1: Fast filter - only keep blocks with ANY keyword match
    candidate_blocks = []
    progress_placeholder = st.empty()
    
    for item in extract_text_blocks_streaming(uploaded_file):
        # Handle progress updates
        if item.get("type") == "progress":
            progress_placeholder.text(f"📄 Processing pages {item['current']}/{item['total']}...")
            continue
        
        text_lower = item["text"].lower()
        
        # Quick check: does this block contain ANY keyword?
        if any(k in text_lower for k in keywords):
            candidate_blocks.append(item)
    
    progress_placeholder.empty()
    
    # If no matches found at all
    if not candidate_blocks:
        return []
    
    # PASS 2: Detailed scoring ONLY on candidates (much smaller set)
    scored = []
    for block in candidate_blocks:
        text_lower = block["text"].lower()
        # Count total keyword frequency (better than just presence)
        score = sum(text_lower.count(k) for k in keywords)
        scored.append((score, block))
    
    # Sort by score (highest first) and return top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in scored[:max_blocks]]


# =================================================
# CONFIDENCE CALCULATION
# =================================================
def calculate_confidence(block_count, total_keywords_found):
    """Calculate answer confidence based on evidence strength"""
    if block_count >= 8 and total_keywords_found > 15:
        return "🟢🟢 High Confidence", "Strong evidence found across multiple sections"
    elif block_count >= 3 and total_keywords_found > 5:
        return "🟢 Medium Confidence", "Relevant sections found"
    else:
        return "🟡 Low Confidence", "Limited evidence available"


# =================================================
# STREAMLIT SETUP
# =================================================
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Document Assistant")
st.caption("Optimized for large documents (500-1000+ pages). Ask questions, get cited answers.")


# =================================================
# ENHANCED SYSTEM PROMPT
# =================================================
SYSTEM_PROMPT = """You are a STRICT Document Evidence Interpreter.

CRITICAL RULES:
1. Answer ONLY using the provided document excerpts below
2. ALWAYS cite Block IDs in your answer (e.g., "According to Block B42 on page 15...")
3. If information appears contradictory across blocks, explicitly mention this
4. If the answer is not explicitly present in the excerpts, respond EXACTLY with:
   "This is not covered in the document."
5. Be concise but complete - include all relevant details from the blocks
6. Never add information from your training data - stick strictly to the document

FORMAT:
- Start with a direct answer
- Support with specific Block ID citations
- Keep response focused and clear
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
    st.error(f"⚠️ OpenAI client initialization failed: {str(e)}")
    st.stop()


# =================================================
# SIDEBAR CONFIGURATION
# =================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload a text-based PDF (not scanned images)"
    )
    
    max_blocks = st.slider(
        "Maximum context blocks",
        min_value=6,
        max_value=20,
        value=12,
        help="More blocks = more context but slower processing"
    )
    
    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
    1. **Upload** a PDF (any size)
    2. **Ask** a specific question
    3. **Get** answers with block citations
    
    ⚡ **Optimized for:**
    - Documents up to 1000+ pages
    - Fast keyword-based search
    - Memory-efficient processing
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use specific keywords
    - Ask focused questions
    - Check cited blocks for context
    """)


# =================================================
# MAIN INPUT AREA
# =================================================
query = st.text_area(
    "Ask a question about the document",
    height=100,
    placeholder="Example: What are the key findings about climate change impacts?"
)


# =================================================
# MAIN PROCESSING BUTTON
# =================================================
if st.button("🔍 Search & Answer", type="primary"):

    # Validation
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF document first.")
        st.stop()
    
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
        st.stop()

    # Step 1: Check document format
    with st.spinner("Checking document format..."):
        if not document_suitability_check(uploaded_file):
            st.error("⚠️ This PDF appears to be scanned or image-based.")
            st.info("💡 This tool works with text-based PDFs. Please use OCR software to convert scanned PDFs first.")
            st.stop()

    # Step 2: Search for relevant blocks
    st.markdown("### ⏳ Searching document...")
    
    try:
        context_blocks = smart_filter_blocks(uploaded_file, query, max_blocks)
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        st.stop()

    # Step 3: Handle no results
    if not context_blocks:
        st.warning("❌ No relevant sections found for your query.")
        st.info("💡 **Try:**\n- Using different keywords\n- Rephrasing your question\n- Being more specific")
        st.stop()

    # Step 4: Build context for LLM
    context_text = "\n\n".join(
        f"Block ID: {b['id']}\nPage: {b['page']}\nContent:\n{b['text']}"
        for b in context_blocks
    )

    # Step 5: Generate answer with LLM
    with st.spinner("Generating answer from document..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"""Document excerpts:

{context_text}

Question: {query}
"""
                    }
                ],
                temperature=0.2,
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

    # Handle "not covered" response
    if "not covered in the document" in answer.lower():
        st.info("💡 **Suggestions:**\n- Try different keywords\n- Rephrase your question\n- Check if the topic exists in the document")
        st.stop()

    # Calculate and display confidence
    st.markdown("---")
    total_keyword_matches = sum(
        sum(b["text"].lower().count(k) for k in query.lower().split())
        for b in context_blocks
    )
    
    confidence_level, confidence_desc = calculate_confidence(
        len(context_blocks),
        total_keyword_matches
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Confidence", confidence_level)
    with col2:
        st.info(f"**{confidence_desc}**\n\nBased on {len(context_blocks)} relevant blocks with {total_keyword_matches} keyword matches")

    # Evidence viewer
    st.markdown("---")
    with st.expander("📚 View Evidence Blocks from Document", expanded=False):
        st.caption("These are the sections used to generate the answer above")
        
        for idx, b in enumerate(context_blocks, 1):
            st.markdown(f"**Block {b['id']}** | Page {b['page']} | Section {idx}/{len(context_blocks)}")
            
            # Show preview with expand option
            preview_text = b["text"][:500]
            if len(b["text"]) > 500:
                st.text(preview_text + "...")
                if st.button(f"Show full text", key=f"expand_{b['id']}"):
                    st.text(b["text"])
            else:
                st.text(preview_text)
            
            st.markdown("---")

    # Footer stats
    st.markdown("---")
    st.caption(f"✅ Analyzed {len(context_blocks)} most relevant sections | 🔍 Keywords: {', '.join(query.lower().split()[:5])}")


# =================================================
# EMPTY STATE
# =================================================
if not uploaded_file:
    st.info("👈 Upload a PDF from the sidebar to get started")
    
    with st.expander("📋 Example Questions"):
        st.markdown("""
        - What are the main conclusions?
        - What methodology was used?
        - What are the key findings about [topic]?
        - What recommendations are provided?
        - What are the limitations mentioned?
        """)
