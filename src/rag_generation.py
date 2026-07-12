# Import all necessary libraries
import pandas as pd
import time
from src.eval_config import *
from src.eval import generate_rag_responses
from rag_pipeline.query_engine import vectorstore

df = pd.read_json(QA_TESTSET_PATH, orient="records", lines=True)

test_set = df.sample(n=50, random_state=0)

vectorstore_db = vectorstore()

def main(checkpoint_file, k, top_n, search_type):
    generate_rag_responses(test_set, vectorstore_db, file_path=RAG_GENERATION_OUTPUT_PATH, k=RETRIEVAL_K, 
                            top_n=RERANK_TOP_N, search_type=SEARCH_TYPE, session_id="test_session")


if __name__=="__main__":
    print("Starting RAG response generation...")
    main()
    print("RAG response generation completed.")