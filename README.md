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

DocQuest is a simple and interactive Streamlit web app that allows users to ask questions from uploaded documents and receive relevant answers using information retrieval techniques.

## 🚀 Features

- Upload documents in **PDF, TXT, or DOCX** format
- Ask natural language questions related to the uploaded document
- Real-time **question answering** powered by embeddings and a language model
- Displays chat history of previously asked questions and answers
- Intuitive and lightweight UI with branding support


## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain / Gemini (embedding & LLM API)
- Document ingestion & text extraction
- Session state for chat history

## 📁 Directory Structure

```bash
QAWithPDF/
├── data_ingestion.py      # Loads and parses uploaded documents
├── embedding.py           # Generates document embeddings using Gemini
├── model_api.py           # Loads the LLM for answering questions
StreamlitApp.py            # Main Streamlit app script
logo.png                   # App logo
README.md
requirements.txt
```