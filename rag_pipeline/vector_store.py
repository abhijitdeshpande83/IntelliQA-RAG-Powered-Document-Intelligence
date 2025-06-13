import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_db(texts, persist_directory,embedding=embeddings):
    if not os.path.exists(persist_directory):
        print("Creating vector space")
        vectorstore =  Chroma.from_texts(texts=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
        

    else:
        print("Adding text to existing vector space")
        vectorstore = Chroma(persist_directory=persist_directory,
                                embedding_function=embedding
                                )
        vectorstore.add_texts(texts=texts)

    return vectorstore
        
def load_vector_db(persist_directory, embedding_function=embeddings):
        print("Loading existing vector space")
        vectorstore = Chroma(persist_directory=persist_directory,
                            embedding_function=embeddings)
        return vectorstore