from src.eval import prepare_testset_documents
from rag_pipeline.query_engine import vectorstore
from src.eval_config import TEST_DATA_PATH, CHUNK_SIZE, OVERLAP, BATCH_SIZE 

def main():
    docs = prepare_testset_documents(TEST_DATA_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    print(f"Successfully prepared {len(docs)} documents.")

    print(f"Creating vectorstore for {len(docs)} documents...")
    vectorstore(documents=docs, batch_size=BATCH_SIZE)
    print("Document vectorstore created successfully.")

if __name__ == "__main__":
    main()