import os
from typing import List
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from rag_pipeline.models import get_reranker, get_pool_connection
from rag_pipeline.config import RetrievalConfig
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

pool = get_pool_connection()

class PostgresFTSRetriever(BaseRetriever):
    """
    PostgreSQL full-text search retriever for session-scoped document retrieval.

    Uses PostgreSQL tsvector search to retrieve keyword-relevant documents
    filtered by session_id.
    """
    session_id: str
    k: int
    connection_string: str = os.getenv("DATABASE_URL")

    def _get_relevant_documents(self, query, *, run_manager=None) -> List[Document]:
        sql="""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE cmetadata->>'session_id' = %s
              AND content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(content_tsv, plainto_tsquery('english', %s)) DESC
            LIMIT %s
        """
        with pool.connection() as conn:
            rows = conn.execute(sql, (self.session_id, query, query, self.k)).fetchall()
        return [Document(page_content=r[0], metadata=r[1]) for r in rows]


def get_dense_retriever(vectorstore, session_id, config:RetrievalConfig):
    """
    Create a session-filtered dense vector retriever using PGVector.
    """
    return vectorstore.as_retriever(
                    search_type= config.search_type,
                    search_kwargs={
                        "filter": {"session_id": session_id},
                        "k": config.k
                        })
        
def get_retriever(vectorstore, session_id, config:RetrievalConfig):
    """
    Build the RAG retrieval pipeline.

    Supports dense-only or hybrid retrieval using PGVector dense search
    and PostgreSQL full-text search, with optional reranking.

    Args:
        vectorstore: PGVector store instance.
        session_id: Session identifier used to filter documents.
        config: Retrieval configuration containing retrieval size,
            reranking settings, search mode, and hybrid weights.

    Returns:
        Configured retriever with optional hybrid retrieval and reranking.
    """
     
    dense_retriever = get_dense_retriever(vectorstore, session_id, config=config)

    if config.hybrid:
          sparse_retriever = PostgresFTSRetriever(session_id=session_id,k=config.k)
          base = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever], weights=config.hybrid_weights)
    else:
        base = dense_retriever
    
    if not config.rerank:
        return base

    return ContextualCompressionRetriever(
                                base_retriever=base,
                                base_compressor=get_reranker(top_n=config.top_n),
                                )