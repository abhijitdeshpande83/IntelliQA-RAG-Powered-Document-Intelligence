import os
import re
import warnings
from tika import parser
from rag_pipeline.utils import parse, supported_file_types, get_file_extension
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from rag_pipeline.vector_store import get_vector_store
from rag_pipeline.config import *
from rag_pipeline.models import get_llm, get_reranker
warnings.filterwarnings("ignore")

llm = get_llm()

def is_supported_file(file):
    """
    Checks whether a file type is supported for ingestion.

    Args:
        file (str): File path.

    Returns:
        bool: True if file extension is supported, else False.
    """

    return get_file_extension(file) in supported_file_types()

def load_data(file_path, session_id, file_name, chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP):
    """
    Loads a file, extracts text, and splits it into chunks for indexing.

    Args:
        file_path (str): Path to input file.
        session_id (str): Session identifier for filtering in vector DB.
        file_name (str): Original file name for metadata.
        chunk_size (int, optional): Size of each text chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 50.

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
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                    )
    
    chunks = text_splitter.split_documents([doc])
    return chunks

def vectorstore(documents=None, batch_size=BATCH_SIZE):
    """
    Creates or loads a vector store for retrieval.

    Args:
        documents (list, optional): Documents to index.
        batch_size (int, optional): Number of documents to process in each batch. Defaults to 1000. 
    Returns:
        VectorStore: Initialized or loaded vector database.
    """

    if documents:
        return get_vector_store(documents=documents, batch_size=batch_size)
    else:
        return get_vector_store()


def ask_question(question, vectorstore, session_id, return_metadata=RETURN_METADATA, 
                    k=RETRIEVAL_K, search_type=SEARCH_TYPE, top_n=RERANK_TOP_N):
    """
    Runs a Retrieval-Augmented Generation (RAG) pipeline for a query.

    Retrieves relevant documents filtered by session_id and generates
    a response using an LLM. Optionally returns the complete retrieval
    output for evaluation purposes.

    Args:
        question (str): Input query.
        vectorstore: Vector database for retrieval.
        session_id (str): Session filter for retrieval.
        return_metadata (bool, optional): Return full pipeline output instead of only the answer. Defaults to False.
        k (int, optional): Number of documents to retrieve. Defaults to 4.
        search_type (str, optional): Retrieval strategy (e.g., "similarity", "mmr"). Defaults to "similarity".
        top_n (int, optional): Number of top documents to return after reranking. Defaults to 5.

    Returns:
        str | dict:
            - If return_metadata=False, returns the generated answer.
            - If return_metadata=True, returns the complete RetrievalQA response.
    """

    retriever = vectorstore.as_retriever(
                    search_type= search_type,
                    search_kwargs={
                        "filter": {"session_id": session_id},
                        "k": k
                        }
                    )
    prompt_template = """You are a helpful assistant answering questions based on the provided context.

    Use the information in the context below to answer the question. The context may contain the answer directly or in pieces you need to connect. Read it carefully and use any relevant information you find, even if it is partial or phrased differently from the question.

    Only respond that you cannot answer if the context contains nothing relevant to the question. Do not refuse simply because the answer is not stated word-for-word.

    Context:
    {context}

    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
        )

    compressor = get_reranker(top_n=top_n)

    compression_retriever = ContextualCompressionRetriever(
                                base_retriever=retriever,
                                base_compressor=compressor,
                                )

    pipeline = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=compression_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
                )
    response = pipeline.invoke(question)
    return response if return_metadata else response['result']

