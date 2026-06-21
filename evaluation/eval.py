import os
from os import path
import re
import json
from datetime import datetime
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from ragas import RunConfig
from ragas.testset import TestsetGenerator   
load_dotenv()
from rag_pipeline.query_engine import load_data, ask_question, is_supported_file


def prepare_testset_documents(eval_data_path):
    """
    Loads and prepares documents from a directory for synthetic QA generation.

    Traverses all files in the given path, filters supported file types,
    and loads them into LangChain-compatible document objects.

    Args:
        eval_data_path (str): Path to directory containing source documents.

    Returns:
        list: List of loaded document objects.
    """

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

def generate_qa_dataset(docs, generator_llm, generator_embeddings, run_config, test_size):
    """
    Generates a synthetic QA dataset from input documents using Ragas.

    Uses an LLM and embedding model to create question-answer pairs
    for evaluation of a RAG pipeline.

    Args:
        docs (list): Input documents.
        generator_llm: LLM wrapper used for QA generation.
        generator_embeddings: Embedding model wrapper.
        run_config (RunConfig): Execution configuration for generation.
        test_size (int): Number of QA pairs to generate.

    Returns:
        Dataset: Generated synthetic QA dataset.
    """       
    
    generator = TestsetGenerator(
                                llm=generator_llm, 
                                embedding_model=generator_embeddings
                                 )

    return generator.generate_with_langchain_docs(docs, testset_size=test_size, run_config=run_config)

def get_rag_response(question: str, vectorstore_db, session_id: str) -> dict:
    """
    Executes the RAG pipeline for a single query.

    Retrieves relevant context from the vector store and generates an
    LLM-based response.

    Args:
        question (str): User query.
        vectorstore_db: Vector database for retrieval.
        session_id: Session identifier for tracing.

    Returns:
        dict: Contains generated answer and retrieved contexts.
    """

    response = ask_question(question, vectorstore_db, session_id, eval=True)
    
    answer = response['result']
    contexts = [doc.page_content for doc in response['source_documents']]
    
    return {"answer": answer, "contexts": contexts}

def save_checkpoints(data, file_path):
    """
    Appends a single record to a JSONL checkpoint file.

    Each record is stored as a separate JSON line for incremental
    saving and crash recovery.

    Args:
        data (dict): Single record to save.
        file_path (str): Path to JSONL file.
    """
    
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')

def generate_rag_responses(df, vectorstore_db, session_id):
    """
    Runs the RAG pipeline over a dataset and stores outputs as JSONL.

    Iterates over input questions, generates responses using the RAG system,
    and saves each result incrementally for checkpointing and recovery.

    Args:
        df (DataFrame): Input dataset containing questions and references.
        vectorstore_db: Vector database for retrieval.
        session_id: Session identifier for RAG tracking.

    Returns:
        None
    """

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            response = get_rag_response(row.user_input, vectorstore_db, session_id)

            result = {
                "user_input": row.user_input,
                "retrieved_contexts": response["contexts"],
                "response": response["answer"],
                "reference": row.reference
            }

            save_checkpoints(result, "test_data/rag_results.jsonl")
            print(f"Row {i} processed and saved.")

        except Exception as e:
            print("Current time:", datetime.now().strftime("%H:%M:%S"))
            print("Try after ", re.search(r'(\d+m+\d.)', str(e)).group(0))
            print(f"Error processing row {i}: {e}")
            break

