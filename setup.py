from setuptools import setup, find_packages

setup(
    name='rag_pipeline',
    version='3.0',
    packages=find_packages(),
    install_requires=[
        "langchain",
        "tika",
        "python-dotenv",
        "langchain-chroma",
        "langchain-core",
        "langchain-text-splitters",
        "langchain-openai",
        "langchain-community",
        "langchain-experimental",
        "langchain-huggingface",
        "sentence-transformers",
        ],
        python_requires='>=3.10',
        author='Abhijit Deshpande',
        description='RAG Pipeline for IntelliQA project - add session_id metadata to document chunks and enable retriever filtering by session_id'
    )