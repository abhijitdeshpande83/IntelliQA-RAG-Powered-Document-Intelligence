from rag_pipeline.models import get_pgvectore_connection

def get_vector_store(documents=None, batch_size=1000):
    """
    Initializes the PGVector store and adds documents if provided.
    """
    vectorestore = get_pgvectore_connection()

    if documents:
        print(f"Adding {len(documents)} documents to PGVector...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorestore.add_documents(documents=batch)
        print("Successfully added documents to PGVector.")
    return vectorestore