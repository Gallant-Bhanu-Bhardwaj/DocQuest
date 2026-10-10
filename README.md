---
title: DocQuest
emoji: 📄
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
---

# 📄 DocQuest – QA with Documents

![DocQuest logo](logo.png)

DocQuest is a Streamlit app for asking questions about your own documents. It runs a small
retrieval-augmented generation (RAG) pipeline: documents are chunked and embedded locally,
the most relevant chunks are retrieved with FAISS, and an LLM answers from that context only,
with citations back to the source file and page.

## 🚀 Features

- Upload **PDF, TXT, or DOCX** files (multiple at once)
- **OCR fallback**: scanned PDFs with no text layer are rasterized and read with Tesseract
- Answers grounded in the retrieved context, with **citations** (file name + page)
- **Retrieval panel** showing the exact chunks the model saw, with distance scores
- **Edit answers**: save a correction and it is reused for near-duplicate questions
- Short conversation memory (last few turns) for follow-up questions

## 🛠️ Stack

| Concern | Choice |
| --- | --- |
| UI | Streamlit |
| Text extraction | PyPDF2 (PDF), python-docx (DOCX), plain decode (TXT) |
| OCR fallback | pdf2image + Tesseract (pytesseract), OpenCV Otsu thresholding |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers / langchain-huggingface (local, cached in `models/`) |
| Vector store | FAISS (in-memory, top-3 similarity search) |
| LLM | OpenAI `gpt-4o-mini` via langchain-openai |

## ⚙️ Setup

System dependencies (needed for OCR):

- Tesseract OCR (`brew install tesseract` / `apt install tesseract-ocr`)
- Poppler (`brew install poppler` / `apt install poppler-utils`)

Python dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
# Optional: full path to the Tesseract binary if it is not on PATH
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Run:

```bash
streamlit run StreamlitApp.py
```

A `Dockerfile` is included (installs Tesseract and Poppler, serves on port 7860).

## 📁 Layout

```
StreamlitApp.py              # The app: extraction, OCR, chunking, FAISS, QA, UI
logger.py                    # File logging helper
Dockerfile, packages.txt     # Container / Spaces deployment
requirements.txt
legacy-llamaindex-approach/  # Earlier LlamaIndex/Gemini implementation (unused)
```

Runtime folders `models/`, `storage/`, `logs/` are git-ignored.
