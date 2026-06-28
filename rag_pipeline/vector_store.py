import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

CONNECTION_STRING = os.getenv("DATABASE_URL")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
)

def get_vector_store(documents=None, embeddings=embeddings, connection_string=CONNECTION_STRING, batch_size=1000):
    """
    Initializes the PGVector store and adds documents if provided.
    """

    vectorstore = PGVector(
            connection=connection_string,
            embeddings=embeddings,
            collection_name="document_embeddings",
            use_jsonb=True,
            )

    if documents:
        print(f"Adding {len(documents)} documents to PGVector...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(documents=batch)
        print("Successfully added documents to PGVector.")
    return vectorstore
        
