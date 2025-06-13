from setuptools import setup, find_packages

setup(
    name='rag_pipeline',
    version='2.0',
    packages=find_packages(),
    install_requires=[
        'langchain',
        'tika',
        'python-dotenv',
        'langchain_chroma',
        'langchain_core',
        'langchain_text_splitters',
        'langchain_openai',
        'langchain_community',
        'langchain_experimental',
        'langchain_huggingface',
        'sentence_transformers'
        ],
        python_requires='>=3.10',
        author='Abhijit Deshpande',
        description='RAG Pipeline for IntelliQA project - Added remote Tika server support and improved file parsing logic'
    )