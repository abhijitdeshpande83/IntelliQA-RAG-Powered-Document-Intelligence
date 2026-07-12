import os
import re
import json
from datetime import datetime
import random
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from src.eval import prepare_testset_documents
from src.eval_config import *
from rag_pipeline.query_engine import vectorstore

def create_document_vectorstore(docs):
    non_csv_docs = [doc for doc in docs if not doc.metadata["filename"].endswith(".csv")]

    print(f"Excluded {len(docs) - len(non_csv_docs)} CSV documents.")
    print(f"Creating vectorstore for {len(non_csv_docs)} documents...")

    vectorstore(documents=non_csv_docs, batch_size=250)

    print("Document vectorstore created successfully.")

def create_csv_vectorstore(docs):
    csv_docs = [doc for doc in docs if doc.metadata["filename"].endswith(".csv")]

    print(f"Found {len(csv_docs)} CSV documents.")
    print("Creating CSV vectorstore...")

    vectorstore(documents=csv_docs, batch_size=250)

    print("CSV vectorstore created successfully.")


def main():
    docs = prepare_testset_documents(TEST_DATA_PATH, CHUNK_SIZE, OVERLAP)
    print(f"Successfully prepared {len(docs)} documents.")

    create_document_vectorstore(docs)
    create_csv_vectorstore(docs)

if __name__ == "__main__":
    main()