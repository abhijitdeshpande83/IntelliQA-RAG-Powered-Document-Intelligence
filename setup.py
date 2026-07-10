from setuptools import setup, find_packages

setup(
    name='rag_pipeline',
    version='3.3',
    packages=find_packages(),
    install_requires=[
    "langchain>=0.3.30,<0.4",
    "langchain-core>=1.4,<2.0",
    "langchain-community>=0.3.31,<0.4",
    "langchain-text-splitters>=1.1,<2.0",
    "langchain-classic>=1.0,<2.0",
    "langchain-huggingface>=0.3,<1.0",
    "langchain-postgres>=0.0.15,<1.0",  # PGVector
    "langchain-groq>=0.2,<0.3",
    "sentence-transformers>=2.2",
    "psycopg[binary]>=3.2",             # PostgreSQL driver
    "tika>=2.6",
    ],
    python_requires='>=3.10',
    author='Abhijit Deshpande',
    description="RAG Pipeline for IntelliQA project providing retrieval-augmented generation " \
    "capabilities with document ingestion, vector search, and LLM-based question answering.",
)