import os
from os import path
import subprocess
from pathlib import Path
from dotenv import load_dotenv
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

    