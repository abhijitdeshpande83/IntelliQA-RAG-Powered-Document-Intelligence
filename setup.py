from setuptools import setup, find_packages

setup(
    name='rag_pipeline',
    version='3.2',
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.30,<0.4",
        "langchain-core>=1.4,<2.0",
        "langchain-community>=0.3.31,<0.4",
        "langchain-text-splitters>=1.1,<2.0",
        "langchain-classic>=1.0,<2.0",          # provides RetrievalQA on 1.x
        "langchain-huggingface>=0.3,<1.0",
        "langchain-chroma>=0.2,<0.3",
        "langchain-groq>=0.2,<0.3",
        "sentence-transformers>=2.2",
        "tika>=2.6",
        "python-dotenv>=1.0"
    ],
    python_requires='>=3.10',
    author='Abhijit Deshpande',
    description='RAG Pipeline for IntelliQA project - add session_id metadata to document chunks and enable retriever filtering by session_id',
)