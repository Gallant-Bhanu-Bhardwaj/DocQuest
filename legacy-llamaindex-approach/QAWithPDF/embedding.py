from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # ✅ main fix
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from QAWithPDF.exception import customexception
from logger import logging


def download_local_embedding(model, document):
    """
    Builds and initializes a local embedding model for vector embeddings (no API required).

    Returns:
    - query_engine: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("Initializing local HuggingFace embedding model...")

        local_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        Settings.llm = model
        Settings.embed_model = local_embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        logging.info("Creating local vector index...")
        index = VectorStoreIndex.from_documents(document)

        logging.info("Local embedding index successfully created.")
        query_engine = index.as_query_engine()

        return query_engine

    except Exception as e:
        raise customexception(e, sys)
