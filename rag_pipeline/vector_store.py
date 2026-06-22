import os
from click.core import batch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_db(documents, persist_directory, embedding=embeddings, batch_size=1000):
    """
    Creates or updates a Chroma vector database from documents.

    If the vector store does not exist, it is created from scratch.
    If it already exists, new documents are added in batches.

    Args:
        documents (list): Documents to embed and store.
        persist_directory (str): Path to persist vector database.
        embedding: Embedding model used for vectorization.
        batch_size (int): Number of documents to process per batch.

    Returns:
        Chroma: Initialized or updated vector store instance.
    """
    
    if not os.path.exists(persist_directory):
        print("Creating vector space")
        vectorstore =  Chroma.from_documents(documents=documents,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
        

    else:
        print("Adding documents to existing vector space")
        vectorstore = Chroma(persist_directory=persist_directory,
                                embedding_function=embedding
                                )
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(documents=batch)

    return vectorstore
        
def load_vector_db(persist_directory, embedding_function=embeddings):
    """
    Loads an existing Chroma vector database from disk.

    Args:
        persist_directory (str): Path to stored vector database.
        embedding_function: Embedding model used for retrieval.

    Returns:
        Chroma: Loaded vector store instance.
    """
    
    print("Loading existing vector space")
    vectorstore = Chroma(persist_directory=persist_directory,
                        embedding_function=embeddings)
    return vectorstore