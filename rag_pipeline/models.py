import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_postgres import PGVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline.config import *
load_dotenv()

@lru_cache(maxsize=1)
def get_pgvectore():
    """
    Initializes and returns a cached instance of the PGVector store.

    Returns:
        PGVector: Configured PGVector instance.
    """
    return  PGVector(
            connection=os.getenv("DATABASE_URL"),
            embeddings=get_embeddings(),
            collection_name=COLLECTION_NAME,
            use_jsonb=True,
            )

@lru_cache(maxsize=1)
def get_llm():
    """
    Initializes and returns a cached instance of the LLM.

    Returns:
        ChatGroq: Configured LLM instance.
    """
    return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE)

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Initializes and returns a cached instance of the embeddings model.

    Returns:
        HuggingFaceEmbeddings: Configured embeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

@lru_cache(maxsize=1)
def get_cross_encoder():
    """
    Initializes and returns a cached instance of the reranker model.

    Returns:
        HuggingFaceCrossEncoder: Configured reranker instance.
    """
    return HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL
        )

@lru_cache(maxsize=1)
def get_reranker(top_n=RERANK_TOP_N):
    """
    Initializes and returns a cached instance of the CrossEncoderReranker.

    Returns:
        CrossEncoderReranker: Configured reranker instance.
    """
    return CrossEncoderReranker(
                    model=get_cross_encoder(),
                    top_n=RERANK_TOP_N
                    )   