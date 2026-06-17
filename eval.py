import os
from os import path
import subprocess
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from rag_pipeline.query_engine import load_data, vectorstore, ask_question, is_supported_file


def ingest_documents(evaluation_data_path):

    """
    Recursively loads supported documents from the specified directory
    and ingests them into the vector store.
    """

    path = Path(evaluation_data_path)
    files = [str(file) for file in path.rglob("*") if file.is_file()]

    print("Ingesting documents...")
    for file_path in files:
        file_name = os.path.basename(file_path)
       
        if is_supported_file(file_path):
            session_id = 'test_session'
            docs = load_data(file_path=file_path, session_id=session_id, file_name=file_name)

            vectorstore_db = vectorstore(persist_directory="/Volumes/LaCie/Projects_portfolio/IntelliQA/chroma_db", 
                                     documents=docs)
            # subprocess.run(["./cu.sh"])
    
    print(f"Ingested {len(files)} files into the vector store.")


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

if __name__ == "__main__":
    pass