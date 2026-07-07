# Import all necessary libraries
import pandas as pd
from eval import (prepare_testset_documents, 
                  generate_rag_responses, 
                  run_batch_evaluation,
                  get_score, save_file, 
                  get_results
                  )
from rag_pipeline.query_engine import vectorstore

df = pd.read_json("evaluation/testset/new_testset_rewritten.jsonl", orient="records", lines=True)

test_set = df.sample(n=50, random_state=0)

vectorstore_db = vectorstore()

def main():
    generate_rag_responses(test_set, vectorstore_db, session_id="test_session",
                           checkpoint_file="evaluation/rag_results/rag_results_v4.jsonl",
                           k=20)


if __name__=="__main__":
    main()