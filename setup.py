from setuptools import setup, find_packages

setup(
    name='rag_pipeline',
    version='3.0',
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.27,<0.4",
        "langchain-core>=0.3.78,<1.0",
        "langchain-community>=0.3.29,<0.4",
        "langchain-text-splitters>=0.3.11,<0.4",
        "langchain-openai>=0.3.35,<0.4",
        "langchain-experimental>=0.3,<0.4",
        "langchain-huggingface>=0.3,<1.0",
        "langchain-chroma>=0.2,<0.3",
        "langchain-groq>=0.2,<0.3",
        "sentence-transformers>=2.2",
        "tika>=2.6",
        "python-dotenv>=1.0",
    ],
    python_requires='>=3.10',
    author='Abhijit Deshpande',
    description='RAG Pipeline for IntelliQA project - add session_id metadata to document chunks and enable retriever filtering by session_id',
)