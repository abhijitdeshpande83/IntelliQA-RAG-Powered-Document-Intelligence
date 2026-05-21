import os
import re
from tika import parser
import warnings
from rag_pipeline.utils import parse
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from rag_pipeline.vector_store import create_vector_db, load_vector_db
warnings.filterwarnings("ignore")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))


def load_data(file_path, session_id, file_name):
    # path = os.getcwd()+'/data'
    # files = get_file(path)
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
    return response['result']

