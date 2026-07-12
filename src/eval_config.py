LLM_JUDGE = "gemini-flash-lite-latest"
LLM_GENERATOR = "qwen2.5"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

TEMPERATURE = 0
CHUNK_SIZE = 1000
OVERLAP = 50
BATCH_SIZE = 1000
RETRIEVAL_K = 20
RERANK_TOP_N = 8
SEARCH_TYPE = "similarity"
RETURN_METADATA = True

QA_TESTSET_PATH = "evaluation/testset/new_testset_rewritten.jsonl"

RAG_GENERATION_OUTPUT_PATH = "evaluation/rag_results/rag_results_v6.jsonl"
RAG_EVALUATION_OUTPUT_PATH = "evaluation/eval_results/rag_eval_v6.jsonl"

MAX_WORKERS = 1
TIMEOUT = 180
