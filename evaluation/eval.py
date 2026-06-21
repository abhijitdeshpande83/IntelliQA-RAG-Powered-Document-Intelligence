import os
from os import path
import re
import pandas as pd
from datetime import datetime
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from ragas import RunConfig
from ragas.testset import TestsetGenerator   
load_dotenv()
from rag_pipeline.query_engine import load_data, vectorstore, ask_question, is_supported_file




def prepare_testset_documents(eval_data_path):

    path = Path(eval_data_path)
    files = [str(file) for file in path.rglob("*") if file.is_file()]

    docs = []

    for file_path in files:
        if is_supported_file(file_path):
            session_id = 'test_session'
            file_name = os.path.basename(file_path)
            docs.extend(
                load_data(file_path, session_id, file_name)
            )

    return docs

def generate_qa_dataset(docs, generator_llm, generator_embeddings,run_config, test_size):        

    generator = TestsetGenerator(
                                llm=generator_llm, 
                                embedding_model=generator_embeddings
                                 )

    return generator.generate_with_langchain_docs(docs, testset_size=test_size, run_config=run_config)

def get_rag_response(question: str, vectorstore_db, session_id: str)->dict:

    response = ask_question(question, vectorstore_db, session_id, eval=True)
    
    answer = response['result']
    contexts = [doc.page_content for doc in response['source_documents']]
    
    return {"answer": answer, "contexts": contexts}

def save_checkpoints(data, file_path):

    new_results = pd.DataFrame(data)

    if os.path.exists(file_path):
        old_data = pd.read_json(file_path, orient="records")
        combined_results = pd.concat([old_data, new_results], ignore_index=True)
    else:
        combined_results = new_results

    combined_results.to_json(file_path, orient="records", indent=2)

def build_data(df, vectorstore_db, session_id):
    data = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            response = get_rag_response(row.user_input, vectorstore_db, session_id)

            data.append({
                "user_input": row.user_input,
                "retrieved_contexts": response["contexts"],
                "response": response["answer"],
                "reference": row.reference
            })

            save_checkpoints(data, "test_data/rag_results.json")
            print(f"Row {i} processed and saved.")

        except Exception as e:
            print("Current time:", datetime.now().strftime("%H:%M:%S"))
            print("Try after ", re.search(r'(\d+m+\d.)', str(e)).group(0))
            print(f"Error processing row {i}: {e}")
            break
    
    return data
    