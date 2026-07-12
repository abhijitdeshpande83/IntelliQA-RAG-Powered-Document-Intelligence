import os
from os import path
import pandas as pd
import re
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ragas import EvaluationDataset, RunConfig, evaluate
from IPython.display import display, HTML
from ragas.testset import TestsetGenerator   
load_dotenv()
from rag_pipeline.query_engine import load_data, ask_question, is_supported_file


def prepare_testset_documents(eval_data_path, chunk_size=1000, chunk_overlap=50):
    """
    Loads and prepares documents from a directory for synthetic QA generation.

    Args:
        eval_data_path (str): Path to directory containing source documents.
        chunk_size (int, optional): Size of each text chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 50.

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
                load_data(file_path, session_id, file_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            )

    return docs

def generate_qa_dataset(docs, generator_llm, generator_embeddings, run_config, test_size):
    """
    Generates a synthetic QA dataset for RAG evaluation using Ragas.

    Args:
        docs (list): Source documents for synthetic data generation.
        generator_llm: LLM used to generate questions and answers.
        generator_embeddings: Embedding model used for semantic sampling.
        run_config (RunConfig): Configuration for execution settings.
        test_size (int): Number of QA pairs to generate.

    Returns:
        Dataset: Synthetic QA dataset for evaluating RAG pipelines.
    """      
    
    generator = TestsetGenerator(
                                llm=generator_llm, 
                                embedding_model=generator_embeddings
                                 )

    return generator.generate_with_langchain_docs(docs, testset_size=test_size, run_config=run_config)

def get_rag_response(question: str, vectorstore_db, session_id: str, k:int, search_type:str, top_n:int):
    """
    Executes the RAG pipeline for a single query.

    Args:
        question (str): User query.
        vectorstore_db: Vector database for retrieval.
        session_id: Session identifier for tracing.
        k (int): Number of documents to retrieve.
        search_type (str): Retrieval strategy.
        top_n (int): Number of top documents to return after reranking.

    Returns:
        dict: Contains generated answer and retrieved contexts.
    """

    response = ask_question(question, vectorstore_db, session_id, return_metadata=True,
                             k=k, search_type=search_type, top_n=top_n)
    
    answer = response['result']
    contexts = [doc.page_content for doc in response['source_documents']]
    
    return {"answer": answer, "contexts": contexts}

def save_checkpoints(data, file_path):
    """
    Appends records to a JSONL checkpoint file.

    Args:
        data (dict | pd.DataFrame | object): Input record(s) to save.
        file_path (str): Path to the JSONL file.
    """

    if isinstance(data,dict):
        records = [data]
    elif isinstance(data, pd.DataFrame):
        records = data.to_dict(orient='records')
    elif hasattr(data, "to_pandas") and callable(data.to_pandas):
        records = data.to_pandas().to_dict(orient='records')
    else:
        raise ValueError("Unsupported data type for checkpoint saving")

    # Implementation for saving checkpoint
    with open(file_path, 'a') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

def load_processed_inputs(file_path):
        if not os.path.exists(file_path):
            return set()

        processed_inputs=set()
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    processed_inputs.add(json.loads(line)['user_input'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file_path}")
        return processed_inputs

def generate_rag_responses(df, vectorstore_db, session_id, k, search_type, top_n, file_path):
    """
    Runs a RAG pipeline over an input dataset and stores outputs incrementally as JSONL.

    Args:
        df (pd.DataFrame): Input dataset with user queries and references.
        vectorstore_db: Vector store used for document retrieval.
        session_id (str): Session identifier for scoped retrieval.
        k (int, optional): Number of retrieved documents. Defaults to 4.
        search_type (str, optional): Retrieval strategy (e.g., "similarity", "mmr").
            Defaults to "similarity".
        top_n (int, optional): Number of top documents to return after reranking. Defaults to 5.
        file_path (str): Path to JSONL file for incremental saving.

    Returns:
        None
    """    
    processed_inputs = load_processed_inputs(file_path)

    for i, row in enumerate(df.itertuples(index=False), start=1):
        try:
            if row.user_input in processed_inputs:
                continue
            response = get_rag_response(
                row.user_input,
                vectorstore_db,
                session_id,
                k=k,
                search_type=search_type,
                top_n=top_n
            )

            result = {
                "user_input": row.user_input,
                "retrieved_contexts": response["contexts"],
                "response": response["answer"],
                "reference": row.reference
            }

            save_checkpoints(result, file_path)
            processed_inputs.add(row.user_input)
            print(f"Row {i} processed and saved.")

        except Exception as e:
            print("Current time:", datetime.now().strftime("%H:%M:%S"))
            match = re.search(r'(\d+m+\d.)', str(e))
            retry_after = match.group(0) if match else "unknown"
            print("Try after ", retry_after)
            print(f"Error processing row {i}: {e}")
            break

def run_batch_evaluation(rag_results, metrics, run_config, file_path):

    evaluated_inputs = load_processed_inputs(file_path)

    for i, data in enumerate(rag_results, start=1):
        if data['user_input'] in evaluated_inputs:
            continue
        print(f"Processing: {i}")
        result = evaluate(dataset=EvaluationDataset.from_list([data]), metrics=metrics, 
                       run_config=run_config, raise_exceptions=True)
        
        save_checkpoints(result, file_path)
        evaluated_inputs.add(data['user_input'])

def get_score(df):
    if isinstance(df, pd.DataFrame):
        return df[["llm_context_precision_with_reference","context_recall",
                  "faithfulness","answer_relevancy"]].mean()
    else:
        raise TypeError("Input must be a pandas DataFrame")

def save_file(file, file_path):

    try:
        if os.path.exists(file_path):
            return "File already exists please provide another path"
        
        file.to_csv(file_path, index=False)
        print("File saved to successfully")

    except Exception as e:
        print(f"Error saving file: {e}")

def get_results(file_path):

    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        if file_path.endswith('.jsonl'):
            return pd.read_json(file_path, lines=True)
    except Exception as e:
        return f"Error reading results: {e}"

def unanswerable_responses(df):

    mask = (
        df["response"].str.contains("I cannot", case=False, na=False)
        |
        df["response"].str.contains(
        "The context provided does not directly address",
        case=False,
        na=False,
        )
    )
    print(f"Number of unanswerable responses: {len(df[mask])}")

    return df[mask]

def display_context_recall_failures(df, max_height=500):
    subset = df[df['context_recall'] == 0][
        ['user_input', 'retrieved_contexts', 'response', 'reference']
    ]

    print(f"Number of zero context recall cases: {len(subset)}")
    print(f"Indices of unanswerable responses: {subset.index.tolist()}")

    table_html = subset.to_html().replace(
        '<td>',
        '<td style="text-align:left; white-space:pre-wrap; word-wrap:break-word; max-width:400px;">'
    )

    scrollable_html = f"""
    <div style="
        max-height: {max_height}px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
    ">
        {table_html}
    </div>
    """

    display(HTML(scrollable_html))