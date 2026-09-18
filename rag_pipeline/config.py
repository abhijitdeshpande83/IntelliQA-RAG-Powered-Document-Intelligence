from dataclasses import dataclass

LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

TEMPERATURE = 0
CHUNK_SIZE = 1024
OVERLAP = 256
BATCH_SIZE = 250
RETRIEVAL_K = 20
RERANK_TOP_N = 8
SEARCH_TYPE = "similarity"
HYBRID_WEIGHTS = (
                    0.4, # PostgreSQL FTS
                    0.6  # PGVector
                )
HYBRID_SEARCH_ENABLE = True
RERANK_ENABLE=True
RETURN_METADATA = False

COLLECTION_NAME = "document_embeddings"

MAX_TABULAR_ROWS = 5000

@dataclass
class RetrievalConfig:
    k: int = RETRIEVAL_K
    top_n: int = RERANK_TOP_N
    search_type: str = SEARCH_TYPE
    hybrid: bool = HYBRID_SEARCH_ENABLE
    rerank: bool = RERANK_ENABLE
    hybrid_weights: tuple = HYBRID_WEIGHTS
