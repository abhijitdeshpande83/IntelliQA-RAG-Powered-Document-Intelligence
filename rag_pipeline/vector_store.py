import os
from click.core import batch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_db(documents, persist_directory, embedding=embeddings, batch_size=1000):
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
        print("Loading existing vector space")
        vectorstore = Chroma(persist_directory=persist_directory,
                            embedding_function=embeddings)
        return vectorstore