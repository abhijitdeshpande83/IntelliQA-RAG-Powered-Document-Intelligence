import os
import re
from tika import parser
import warnings
from rag_pipeline.utils import parse, supported_file_types, get_file_extension
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag_pipeline.vector_store import create_vector_db, load_vector_db
warnings.filterwarnings("ignore")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))

def is_supported_file(file):
    return get_file_extension(file) in supported_file_types()

def load_data(file_path, session_id, file_name):

    text = parse(file_path)

    doc = Document(
            page_content=text, 
            metadata={
                "session_id": session_id,
                "filename": file_name
                }
            )
    
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=50
                    )
    
    chunks = text_splitter.split_documents([doc])
    return chunks

def vectorstore(persist_directory="chroma_db",documents=None):
    if documents:
        return create_vector_db(persist_directory=persist_directory, documents=documents)
    else:
        return load_vector_db(persist_directory=persist_directory)


def ask_question(Question, vectorstore, session_id):

    retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "filter": {"session_id": session_id}
                        }
                    )

    pipeline = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
                )
    response = pipeline.invoke(Question)
    return response

