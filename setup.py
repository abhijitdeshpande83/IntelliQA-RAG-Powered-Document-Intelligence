from setuptools import setup, find_packages

setup(
    name='rag_pipeline',
    version='3.3',
    packages=find_packages(),
    install_requires=[
    "langchain>=0.3.27,<0.4",
    "langchain-core>=0.3.86,<0.4",
    "langchain-text-splitters>=0.3.11,<0.4",
    "langchain-huggingface>=0.3.1,<0.4",
    "langchain-postgres>=0.0.17,<0.1",
    "langchain-groq>=0.2.5,<0.3",
    "sentence-transformers>=2.2",
    "psycopg[binary]>=3.2",
    "tika>=2.6",
],
    python_requires='>=3.10',
    author='Abhijit Deshpande',
    description="RAG Pipeline for IntelliQA project providing retrieval-augmented generation " \
    "capabilities with document ingestion, vector search, and LLM-based question answering.",
)