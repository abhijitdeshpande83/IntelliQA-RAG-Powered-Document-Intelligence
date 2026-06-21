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
    """
    Checks whether a file type is supported for ingestion.

    Args:
        file (str): File path.

    Returns:
        bool: True if file extension is supported, else False.
    """

    return get_file_extension(file) in supported_file_types()

def load_data(file_path, session_id, file_name):
    """
    Loads a file, extracts text, and splits it into chunks for indexing.

    Steps:
        1. Parses raw file content
        2. Wraps content into a LangChain Document with metadata
        3. Splits document into overlapping chunks for retrieval

    Args:
        file_path (str): Path to input file.
        session_id (str): Session identifier for filtering in vector DB.
        file_name (str): Original file name for metadata.

    Returns:
        list[Document]: Chunked documents ready for embedding/storage.
    """

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

def vectorstore(persist_directory="chroma_db", documents=None):
    """
    Creates or loads a vector store for retrieval.

    If documents are provided, a new vector database is created.
    Otherwise, an existing persisted vector store is loaded.

    Args:
        persist_directory (str): Path to vector DB storage.
        documents (list, optional): Documents to index.

    Returns:
        VectorStore: Initialized or loaded vector database.
    """

    if documents:
        return create_vector_db(persist_directory=persist_directory, documents=documents)
    else:
        return load_vector_db(persist_directory=persist_directory)


def ask_question(Question, vectorstore, session_id, eval=False):
    """
    Runs a Retrieval-Augmented Generation (RAG) pipeline for a query.

    Retrieves relevant documents filtered by session_id and generates
    a response using an LLM. Optionally returns full retrieval context
    for evaluation purposes.

    Args:
        Question (str): Input query.
        vectorstore: Vector database for retrieval.
        session_id (str): Session filter for retrieval.
        eval (bool): If True, returns full response including sources.

    Returns:
        str or dict:
            - If eval=False → generated answer (str)
            - If eval=True → full response dict with sources
    """

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
    return response['result'] if not eval else response

