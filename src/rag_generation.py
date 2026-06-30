# Import all necessary libraries
import pandas as pd
from eval import (prepare_testset_documents, 
                  generate_rag_responses, 
                  run_batch_evaluation,
                  get_score, save_file, 
                  get_results
                  )
from rag_pipeline.query_engine import vectorstore

df = pd.read_json("test_data/curated_rag_evaluation_testset_32.jsonl", orient="records", lines=True)

vectorstore_db = vectorstore()

def main():
    generate_rag_responses(df, vectorstore_db, session_id="test_session", k=20)
    

if __name__=="__main__":
    main()