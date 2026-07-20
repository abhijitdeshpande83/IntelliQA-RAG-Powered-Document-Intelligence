from dataclasses import dataclass

LLM_JUDGE = "gemini-flash-lite-latest"
LLM_GENERATOR = "qwen2.5"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

TEMPERATURE = 0
CHUNK_SIZE = 512
OVERLAP = 128
BATCH_SIZE = 250
RETRIEVAL_K = 20
RERANK_TOP_N = 8
SEARCH_TYPE = "similarity"
HYBRID_WEIGHTS = (
                    0.4, # PostgreSQL FTS
                    0.6  # PGVector
                )
HYBRID_SEARCH_ENABLE = False
RERANK_ENABLE=True
RETURN_METADATA = True

COLLECTION_NAME = "document_embeddings"

@dataclass
class RetrievalConfig:
    k: int = RETRIEVAL_K
    top_n: int = RERANK_TOP_N
    search_type: str = SEARCH_TYPE
    hybrid: bool = HYBRID_SEARCH_ENABLE
    rerank: bool = RERANK_ENABLE
    hybrid_weights: tuple = HYBRID_WEIGHTS


TEST_DATA_PATH = "evaluation/eval_data"
QA_TESTSET_PATH = "evaluation/testset/new_testset_rewritten.jsonl"

RAG_GENERATION_OUTPUT_PATH = "evaluation/rag_results/rag_results_v10.jsonl"
RAG_EVALUATION_OUTPUT_PATH = "evaluation/eval_results/rag_eval_v10.jsonl"

MAX_WORKERS = 1
TIMEOUT = 180

TIME_SLEEP=14400