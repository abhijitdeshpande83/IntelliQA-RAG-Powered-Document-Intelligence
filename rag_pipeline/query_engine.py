import os
import re
from tika import parser
import warnings
from rag_pipeline.utils import parse
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_pipeline.vector_store import create_vector_db, load_vector_db
warnings.filterwarnings("ignore")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))


def load_data(file_path):
    # path = os.getcwd()+'/data'
    # files = get_file(path)
    text = parse(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
    return text_splitter.split_text(text)

def vectorstore(persist_directory="chroma_db",texts=None):
    if texts:
        return create_vector_db(persist_directory=persist_directory,texts=texts)
    else:
        return load_vector_db(persist_directory=persist_directory)


def ask_question(Question, vectorstore):
    pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
        return_source_documents=True
        )
    response = pipeline.invoke(Question)
    return response['result']

