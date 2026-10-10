"""Keyword (BM25) and hybrid retrieval, built on top of the existing FAISS store.

Every function works on "keys": a (source, page, text) tuple that identifies one chunk.
"""
from rank_bm25 import BM25Okapi

POOL = 20      # how many candidates each retriever contributes to the fusion
RRF_K = 60     # the standard RRF constant: bigger = flatter, smaller = favours top ranks


def doc_key(doc):
    # A chunk is identified by where it came from plus its text.
    return (doc.metadata["source"], doc.metadata["page"], doc.page_content)


def get_all_docs(vector_store):
    # FAISS (LangChain) keeps every chunk Document in its in-memory docstore.
    return list(vector_store.docstore._dict.values())


def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion.

    rankings: a list of ranked lists (best first), e.g. [dense_keys, bm25_keys].
    Each item scores 1 / (k + rank) in every list it appears in (rank starts at 1);
    the scores are summed and items are returned best-first.
    """
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


def dense_ranking(query, vector_store, pool=POOL):
    # Existing FAISS retriever: nearest chunks by embedding distance, best first.
    results = vector_store.similarity_search_with_score(query, k=pool)
    return [doc_key(doc) for doc, _ in results]


def bm25_ranking(query, docs, pool=POOL):
    # BM25 over the same chunks: lowercase + split on whitespace is the whole tokenizer.
    bm25 = BM25Okapi([d.page_content.lower().split() for d in docs])
    scores = bm25.get_scores(query.lower().split())
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [doc_key(docs[i]) for i in order[:pool]]


def hybrid_ranking(query, vector_store, docs=None):
    # Fuse the dense and BM25 rankings with RRF. Returns keys, best first.
    docs = docs if docs is not None else get_all_docs(vector_store)
    return rrf([dense_ranking(query, vector_store), bm25_ranking(query, docs)])


def hybrid_search(query, vector_store, k=3):
    """Same shape as similarity_search_with_score: [(doc, score)], lower score = better.
    Here the score is simply the fused rank position (1.0, 2.0, ...)."""
    docs = get_all_docs(vector_store)
    by_key = {doc_key(d): d for d in docs}
    top_keys = hybrid_ranking(query, vector_store, docs)[:k]
    return [(by_key[key], float(rank)) for rank, key in enumerate(top_keys, start=1)]
