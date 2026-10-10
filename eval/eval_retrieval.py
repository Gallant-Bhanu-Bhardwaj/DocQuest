"""Retrieval evaluation: FAISS-only vs BM25-only vs hybrid (RRF).

Usage (from the repo root):   venv/bin/python eval/eval_retrieval.py
Reads eval/questions.json:
[{"doc": "file.pdf", "question": "...", "page": 3, "evidence": "...", "type": "keyword"}, ...]
and the PDFs from eval/docs/. Writes the table to eval/results.md.
"""
import json
import os
import sys

from PyPDF2 import PdfReader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from StreamlitApp import build_vector_store   # the app's real pipeline (extract + OCR + chunk + FAISS)
from retrieval import dense_ranking, bm25_ranking, hybrid_ranking, get_all_docs

TOP_K = 3
DOCS_DIR = os.path.join(ROOT, "eval", "docs")


class LocalFile:
    """Minimal stand-in for a Streamlit upload: build_vector_store needs .name and .read()."""
    def __init__(self, path):
        self.name = os.path.basename(path)
        self._path = path

    def read(self):
        with open(self._path, "rb") as f:
            return f.read()


def score(ranked_keys, doc_name, page):
    """Return (hit, reciprocal_rank) for one question, looking only at the top K."""
    for rank, key in enumerate(ranked_keys[:TOP_K], start=1):
        source, chunk_page, _text = key
        if source == doc_name and chunk_page == page:
            return 1, 1.0 / rank      # correct page found at this rank
    return 0, 0.0                     # correct page not in the top K


def main():
    with open(os.path.join(ROOT, "eval", "questions.json")) as f:
        questions = json.load(f)
    if not questions:
        print("eval/questions.json is empty - add questions first.")
        return

    # One vector store per document, so retrieval only searches the document being asked about.
    stores = {}
    for doc_name in sorted({q["doc"] for q in questions}):
        store = build_vector_store([LocalFile(os.path.join(DOCS_DIR, doc_name))])
        if store is None:
            sys.exit(f"Could not index {doc_name}")
        stores[doc_name] = (store, get_all_docs(store))

    methods = {
        "FAISS only": lambda q, store, docs: dense_ranking(q, store),
        "BM25 only": lambda q, store, docs: bm25_ranking(q, docs),
        "Hybrid (RRF k=60)": lambda q, store, docs: hybrid_ranking(q, store, docs),
    }

    # Document name + page count (pages as PyPDF2 counts them, same as the app)
    doc_lines = []
    for doc_name in stores:
        n_pages = len(PdfReader(os.path.join(DOCS_DIR, doc_name)).pages)
        doc_lines.append(f"Document: {doc_name} ({n_pages} pages)")

    lines = doc_lines + [
        f"Questions: {len(questions)}  |  top-{TOP_K} chunks per question",
        "",
        f"| Method | Hit@{TOP_K} | MRR |",
        "|---|---|---|",
    ]
    per_question = [{} for _ in questions]   # per_question[i][method name] = 1 or 0
    for name, retrieve in methods.items():
        hits, rr_total = 0, 0.0
        for i, q in enumerate(questions):
            store, docs = stores[q["doc"]]
            hit, rr = score(retrieve(q["question"], store, docs), q["doc"], q["page"])
            per_question[i][name] = hit
            hits += hit
            rr_total += rr
        n = len(questions)
        lines.append(f"| {name} | {hits}/{n} = {hits / n:.3f} | {rr_total / n:.3f} |")

    # Per-question table: did each method find the correct page in its top K?
    lines += ["", "Per-question results (page = PDF page index, 1-based):", "",
              "| # | Type | Page | " + " | ".join(methods) + " | Question |",
              "|---|---|---|" + "---|" * len(methods) + "---|"]
    for i, q in enumerate(questions):
        marks = " | ".join("hit" if per_question[i][m] else "miss" for m in methods)
        lines.append(f"| {i + 1} | {q.get('type', '')} | {q['page']} | {marks} | {q['question']} |")

    table = "\n".join(lines)
    print(table)
    with open(os.path.join(ROOT, "eval", "results.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
