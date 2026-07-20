from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from rag_pipeline.vector_store import get_vector_store
from rag_pipeline.config import RetrievalConfig, RETURN_METADATA
from rag_pipeline.models import get_llm
from rag_pipeline.retrievers import get_retriever
from rag_pipeline.utils import parse, get_file_extension, SUPPORTED_FORMATS, split_document

llm = get_llm()

def is_supported_file(file):
    """
    Checks whether a file type is supported for ingestion.

    Args:
        file (str): File path.

    Returns:
        bool: True if file extension is supported, else False.
    """

    return get_file_extension(file) in SUPPORTED_FORMATS

def load_data(file_path, session_id, file_name, chunk_size, chunk_overlap):
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

    extension = get_file_extension(file_path)

    text = parse(file_path)
    chunks = split_document(text, extension, chunk_size, chunk_overlap)

    for c in chunks:
        c.metadata.update({"session_id":session_id, "filename":file_name})

    return chunks

def vectorstore(documents, batch_size):
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

def ask_question(question, vectorstore, session_id, config:RetrievalConfig, return_metadata=RETURN_METADATA):
    """
    Runs the RAG pipeline to retrieve relevant session documents and generate an answer.

    Args:
        question (str): User query.
        vectorstore: Vector database instance.
        session_id (str): Session identifier for document filtering.
        config (RetrievalConfig): Retrieval and search configuration.
        return_metadata (bool, optional): Returns full response with sources when True. Defaults to False.

    Returns:
        str | dict: Generated answer or full pipeline response with metadata.
    """
    retriever = get_retriever(vectorstore, session_id, config)

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

    pipeline = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
                )
    response = pipeline.invoke(question)
    return response if return_metadata else response['result']

