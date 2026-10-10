import streamlit as st
import tempfile
import io
import html
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
import os
import sys
import logging

# --- Import OCR & Image Cleaning libraries ---
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import cv2 
import numpy as np 
# ---------------------------------

# --- Import LangChain libraries ---
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from retrieval import hybrid_search
# ---------------------------------
# ---------------------------------

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Optional: point pytesseract at a specific Tesseract binary (otherwise it uses PATH)
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def load_model():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        return model
    except Exception as e:
        st.error("Connection Error: Check that OPENAI_API_KEY is set in your .env file.")
        raise e


# 2. Image Preprocessing
def preprocess_image_for_ocr(image):
    try:
        open_cv_image = np.array(image)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return thresh
    except Exception as e:
        return np.array(image) 

# 3. OCR Extraction
def extract_text_with_ocr(pdf_bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        page_texts = [] 
        for i, img in enumerate(images):
            logging.info(f"Scanning page {i+1}...")
            processed_img = preprocess_image_for_ocr(img)
            text = pytesseract.image_to_string(processed_img) 
            page_texts.append(text) 
        return page_texts 
    except Exception as e:
        logging.error(f"OCR Failed: {e}")
        return [] 

# 4. Main Text Extractor
def extract_text_from_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_files(uploaded_files):
    all_page_data = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_bytes = uploaded_file.read()
        ext = file_name.lower().rsplit(".", 1)[-1]
        logging.info(f"Processing file: {file_name}")

        # A. Plain text and Word docs have no real "pages" - treat as a single page
        if ext == "txt":
            all_page_data.append({"source": file_name, "page": 1, "text": extract_text_from_txt(file_bytes)})
            continue

        if ext == "docx":
            all_page_data.append({"source": file_name, "page": 1, "text": extract_text_from_docx(file_bytes)})
            continue

        # B. PDF: try Fast Digital Extraction
        pdf_bytes = file_bytes
        digital_pages_text = []
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            pdf_reader = PdfReader(tmp_path)
            for page in pdf_reader.pages:
                digital_pages_text.append(page.extract_text() or "")
            os.remove(tmp_path)
        except:
            digital_pages_text = []

        # C. Check if Digital Failed (Scanned PDF)
        full_digital_text = "\n".join(digital_pages_text).strip()

        if not full_digital_text:
            logging.info("No digital text found. Switching to OCR mode...")
            ocr_pages_text = extract_text_with_ocr(pdf_bytes)
            for i, page_text in enumerate(ocr_pages_text):
                all_page_data.append({"source": file_name, "page": i + 1, "text": page_text})
        else:
            for i, page_text in enumerate(digital_pages_text):
                all_page_data.append({"source": file_name, "page": i + 1, "text": page_text})

    return all_page_data

# 5. Chunking
def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

CORRECTION_SIMILARITY_THRESHOLD = 0.8

@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder="models")

def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# 6. Vector Store Builder
def build_vector_store(uploaded_files):
    try:
        all_page_data = extract_text_from_files(uploaded_files)
        if not all_page_data: return None

        embedder = get_embedder()
        texts, metadatas = [], []

        for page_data in all_page_data:
            if not page_data["text"].strip(): continue
            chunks = split_into_chunks(page_data["text"])
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({"source": page_data["source"], "page": page_data["page"]})

        if not texts: return None

        vector_store = FAISS.from_texts(texts, embedder, metadatas=metadatas)
        return vector_store
    except Exception as e:
        logging.error(f"Build failed: {e}")
        return None

# 7. Answer Query
def answer_query(query, vector_store, model, chat_history, edited_answers, embedder, retrieval_mode="Dense"):
    try:
        # 0. Check the corrections store for a near-duplicate question before touching retrieval/the LLM
        if edited_answers:
            query_embedding = embedder.embed_query(query)
            best_score, best_correction = -1.0, None
            for correction in edited_answers:
                score = cosine_similarity(query_embedding, correction["embedding"])
                if score > best_score:
                    best_score, best_correction = score, correction

            if best_correction and best_score >= CORRECTION_SIMILARITY_THRESHOLD:
                logging.info(f"Matched stored correction (similarity={best_score:.2f}) - skipping retrieval/LLM.")
                return best_correction["edited_answer"], []

        if retrieval_mode == "Hybrid":
            # FAISS + BM25 fused with Reciprocal Rank Fusion (see retrieval.py)
            docs_with_scores = hybrid_search(query, vector_store, k=3)
        else:
            docs_with_scores = vector_store.similarity_search_with_score(query, k=3)

        context_parts = []
        retrieved_chunks = []

        for d, score in docs_with_scores:
            src = d.metadata['source']
            pg = d.metadata.get('page', 'N/A')
            context_parts.append(f"From {src} (Page {pg}):\n{d.page_content}")
            retrieved_chunks.append({"source": src, "page": pg, "score": score, "text": d.page_content[:300]})

        context = "\n\n".join(context_parts)

        history_section = ""
        if chat_history:
            recent_turns = chat_history[-4:]
            history_lines = [f"User: {turn['query']}\nAssistant: {turn['answer']}" for turn in recent_turns]
            history_section = "Previous conversation:\n" + "\n\n".join(history_lines) + "\n\n"

        corrections_section = ""
        if edited_answers:
            recent_corrections = edited_answers[-3:]
            correction_lines = [
                f'Note: the user previously corrected a related answer to "{c["query"]}" to say: {c["edited_answer"]}'
                for c in recent_corrections
            ]
            corrections_section = "\n".join(correction_lines) + "\n\n"

        prompt = f"""<s>[INST] You are a helpful assistant. Answer based ONLY on the context below.
If the answer is not in the context, say "I cannot find the answer in the documents."

{history_section}{corrections_section}Context:
{context}

Question: {query} [/INST]"""

        response = model.invoke(prompt).content

        return response, retrieved_chunks
    except Exception as e:
        return f"Error generating answer: {e}", []

# 8. UI Layout (UPDATED WITH TOGGLE EDIT BUTTON)
# --- Design system: fonts, paper-card look, scan-line animation ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

section[data-testid="stMain"] h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.4rem !important;
    display: inline-block;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #4FB6A8;
    margin-bottom: 1.2rem;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Applied explicitly to page numbers, filenames, and score values only */
.qa-mono {
    font-family: 'IBM Plex Mono', monospace;
}

/* Generated answer - plain text on the theme background, sized to stand out */
.qa-answer-text {
    color: #E8E6E1;
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    line-height: 1.7;
    margin: 8px 0 20px 0;
}

/* Retrieved context chunks inside the retrieval transparency expander - plain
   text rows separated by a thin divider, no card container */
.qa-chunk-row {
    padding: 10px 0;
    border-bottom: 1px solid rgba(232, 230, 225, 0.15);
}
.qa-chunk-row:last-child {
    border-bottom: none;
}
.qa-score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 6px;
}
.qa-score-bar-track {
    flex: 1;
    height: 6px;
    background: rgba(232, 230, 225, 0.15);
    border-radius: 4px;
    overflow: hidden;
}
.qa-score-bar-fill {
    height: 100%;
    background: #4FB6A8;
    border-radius: 4px;
}
.qa-chunk-text {
    margin-top: 8px;
    font-size: 0.9rem;
    color: #E8E6E1;
    opacity: 0.75;
}

/* Edit Answer / Save Changes buttons (main content area only - sidebar stays native) */
section[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] {
    background-color: #4FB6A8 !important;
    color: #0E1013 !important;
    border: none !important;
    font-weight: 600 !important;
}
section[data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #3FA294 !important;
    color: #0E1013 !important;
}

/* Edit textarea in mono, so edited text reads as visually distinct from the generated answer */
section[data-testid="stMain"] textarea {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Scan-line animation shown while documents are being indexed */
.qa-scan-wrap {
    position: relative;
    width: 60px;
    height: 76px;
    margin: 6px 0 10px 0;
    border: 1.5px solid #4FB6A8;
    border-radius: 4px;
    overflow: hidden;
    background: rgba(79, 182, 168, 0.06);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
}
.qa-scan-line {
    position: absolute;
    left: 0;
    right: 0;
    height: 2px;
    background: #4FB6A8;
    box-shadow: 0 0 6px #4FB6A8;
    animation: qa-scan-move 1.8s ease-in-out infinite;
}
@keyframes qa-scan-move {
    0%   { top: 2%; }
    50%  { top: 94%; }
    100% { top: 2%; }
}
@media (prefers-reduced-motion: reduce) {
    .qa-scan-line { animation: none; top: 50%; }
}

/* Logo chip - logo.png has an opaque ivory background, so treat it like a small
   paper card (rounded + shadow) instead of a stray square on the dark theme */
[data-testid="stImage"] img {
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
}
section[data-testid="stSidebar"] [data-testid="stImage"] {
    margin-bottom: 12px;
}
</style>
"""

SCAN_ANIMATION_HTML = """
<div class="qa-scan-wrap">
    <div class="qa-scan-line"></div>
    <span>📄</span>
</div>
"""

# logo.png is a detailed wordmark + tagline lockup, not a compact icon mark -
# it turns into an illegible smudge at 16-32px, so the browser tab keeps the emoji.
PAGE_ICON = "📄"
TITLE_TEXT = "AI Document Assistant (OCR Enabled)"

def render_header():
    logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
    logo_loaded = False
    with logo_col:
        try:
            st.image("logo.png", width=52)
            logo_loaded = True
        except Exception as e:
            logging.warning(f"Could not load logo.png for header: {e}")
    with title_col:
        heading = TITLE_TEXT if logo_loaded else f"📄 {TITLE_TEXT}"
        st.markdown(f"<h1>{heading}</h1>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI Document Scanner & QA", page_icon=PAGE_ICON, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_header()
    st.markdown("Upload documents (Text or Scans), ask questions, and **edit the answers** if needed.")

    # --- Session State Initialization ---
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "edited_answers" not in st.session_state:
        st.session_state.edited_answers = []
    if "last_retrieved_chunks" not in st.session_state:
        st.session_state.last_retrieved_chunks = []
    # NEW: Track if we are in "Edit Mode"
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False

    # --- Sidebar: Upload ---
    with st.sidebar:
        try:
            st.image("logo.png", width=36)
        except Exception as e:
            logging.warning(f"Could not load logo.png for sidebar: {e}")
        uploaded_files = st.file_uploader("Upload PDF, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        retrieval_mode = st.radio("Retrieval", ["Dense", "Hybrid"], index=0)
        if uploaded_files and st.button("Process Documents"):
            scan_placeholder = st.empty()
            scan_placeholder.markdown(SCAN_ANIMATION_HTML, unsafe_allow_html=True)
            with st.spinner("Scanning & Indexing... (This may take a moment for OCR)"):
                vs = build_vector_store(uploaded_files)
            scan_placeholder.empty()
            if vs:
                st.session_state.vector_store = vs
                st.success("Index Created Successfully!")

    # --- Main Area: Query & Answer ---
    if st.session_state.vector_store:
        query = st.text_input("Ask a question about your documents:")

        if query:
            last_turn = st.session_state.chat_history[-1] if st.session_state.chat_history else None
            # Only run the AI if the user typed a NEW question
            if last_turn is None or query != last_turn["query"]:
                model = load_model()
                with st.spinner("Analyzing..."):
                    answer, retrieved_chunks = answer_query(query, st.session_state.vector_store, model, st.session_state.chat_history, st.session_state.edited_answers, get_embedder(), retrieval_mode)

                    # Update session state
                    st.session_state.chat_history.append({"query": query, "answer": answer})
                    st.session_state.last_retrieved_chunks = retrieved_chunks
                    st.session_state.edit_mode = False # Reset edit mode on new question

            # --- THE NEW TOGGLE UI ---
            st.markdown("### Answer")

            current_answer = st.session_state.chat_history[-1]["answer"]

            # LOGIC: If Edit Mode is True -> Show Text Area. Else -> Show Markdown.
            if st.session_state.edit_mode:
                # --- EDIT MODE ---
                edited_answer = st.text_area(
                    "Edit the answer below:",
                    value=current_answer,
                    height=300
                )
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("💾 Save Changes"):
                        corrected_query = st.session_state.chat_history[-1]["query"]
                        st.session_state.chat_history[-1]["answer"] = edited_answer
                        corrected_query_embedding = get_embedder().embed_query(corrected_query)
                        st.session_state.edited_answers.append({
                            "query": corrected_query,
                            "edited_answer": edited_answer,
                            "embedding": corrected_query_embedding,
                        })
                        st.session_state.edit_mode = False # Switch back to read-only
                        st.rerun() # Force reload to show the change
            else:
                # --- READ-ONLY MODE ---
                escaped_answer = html.escape(current_answer).replace("\n", "<br>")
                st.markdown(f'<div class="qa-answer-text">{escaped_answer}</div>', unsafe_allow_html=True)
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("✏️ Edit Answer"):
                        st.session_state.edit_mode = True # Switch to edit mode
                        st.rerun() # Force reload to show the text area

            # --- Sources & Retrieval Transparency ---
            if st.session_state.last_retrieved_chunks:
                sources = []
                for chunk in st.session_state.last_retrieved_chunks:
                    citation = f"{chunk['source']} (Page {chunk['page']})"
                    if citation not in sources: sources.append(citation)
                if sources:
                    sources_html = "".join(f'<li><span class="qa-mono">{html.escape(s)}</span></li>' for s in sources)
                    st.markdown(f"**Sources:**<ul>{sources_html}</ul>", unsafe_allow_html=True)

                with st.expander("🔍 Retrieved context (what the model actually saw)"):
                    st.caption("Bar length shows relative relevance within this result set — distance score (lower = more relevant).")
                    scores = [c["score"] for c in st.session_state.last_retrieved_chunks]
                    min_score, max_score = min(scores), max(scores)
                    for chunk in st.session_state.last_retrieved_chunks:
                        # Lower L2 distance = more relevant, so invert when sizing the bar
                        if max_score > min_score:
                            normalized = (chunk["score"] - min_score) / (max_score - min_score)
                        else:
                            normalized = 0.0
                        relevance_pct = max(4, round((1 - normalized) * 100))

                        source_label = html.escape(f"{chunk['source']} (Page {chunk['page']})")
                        chunk_text = html.escape(chunk["text"])

                        st.markdown(f"""
                        <div class="qa-chunk-row">
                            <span class="qa-mono">{source_label}</span>
                            <div class="qa-score-row">
                                <div class="qa-score-bar-track">
                                    <div class="qa-score-bar-fill" style="width:{relevance_pct}%;"></div>
                                </div>
                                <span class="qa-mono">{chunk['score']:.3f}</span>
                            </div>
                            <div class="qa-chunk-text">{chunk_text}...</div>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
