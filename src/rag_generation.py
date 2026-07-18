# Import all necessary libraries
import pandas as pd
import time
from src.eval_config import *
from src.eval import generate_rag_responses
from rag_pipeline.query_engine import vectorstore

df = pd.read_json(QA_TESTSET_PATH, orient="records", lines=True)

test_set = df.sample(n=50, random_state=0)

vectorstore_db = vectorstore()

def main(config:RetrievalConfig):
    generate_rag_responses(test_set, vectorstore_db, file_path=RAG_GENERATION_OUTPUT_PATH, 
                            config=config, session_id="test_session")


if __name__=="__main__":
    print("Starting RAG response generation...")
    main(config=RetrievalConfig())
    print("RAG response generation completed.")