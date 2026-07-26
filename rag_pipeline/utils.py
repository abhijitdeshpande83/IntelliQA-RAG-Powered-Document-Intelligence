import os
import re
import json
from tika import tika
from typing import List
from tika import parser
from langchain_core.documents import Document
from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        HTMLHeaderTextSplitter,
        RecursiveJsonSplitter,
        Language,
        )

# Point tika client to remote server 
tika.TikaClientOnly = True
tika.TikaServerEndpoint = "http://tika:9998"

def get_file_extension(file_name):

    _, ext = os.path.splitext(file_name)

    return ext

def clean_file(text):
    """
    Cleans raw extracted text for RAG pipelines.
    - Removes excessive whitespace
    - Preserves paragraph structure
    """

    text = text.replace('\r\n', '\n').replace('\r', '\n')   # normalize Windows/Mac line endings 
    text = text.replace('\xa0', ' ')                        # non-breaking spaces -> normal space
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)            # rejoin words split across lines

    # 1. REMOVE DOT LEADERS & REPETITIVE PUNCTUATIONS (Fixes TOC noise like ". . . . . . . 14")
    text = re.sub(r'\.{2,}', '', text)                      # Remove blocks of continuous dots
    text = re.sub(r'(\.\s){2,}', '', text)                  # Remove spaced dots like ". . . ."
    text = re.sub(r'_{2,}', '', text)                       # Remove continuous underscores
    text = re.sub(r'(?<![.!?:])\n(?![\n•\-\*\d])', ' ', text)

    # 2. STRIP CLEAN EXTRA WHITESPACE
    text = re.sub(r'[ \t]+', ' ', text)                     # remove excessive spaces/tabs
    text = re.sub(r' *\n *', '\n', text)                    # clean spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)                  # collapse multiple blank lines into two

    return text.strip()

def parse(file):
    """
    Parses a file and extracts its textual content.

    Uses Apache Tika for complex document formats (PDF, DOCX, PPT, etc.)
    and direct file reading for plain text and structured text formats.

    Args:
        file (str): Path to the input file.

    Returns:
        str: Extracted and cleaned text content.
    """

    tika_files = {'.pdf', '.doc', '.docx', '.ppt', '.pptx'}
    html_files = {'.html', '.htm', '.xml', '.md',}
    text_files = {'.txt', '.py', '.ipynb', '.json', '.yaml', '.yml', '.toml'}

    ext = get_file_extension(file)

    if ext in tika_files:
        context = parser.from_file(file)
        # print(f"Successfully parsed {os.path.basename(file)}")
        return clean_file(context['content'].strip())
    
    if ext in html_files:
          with open(file,'r', encoding='utf-8') as f:
                return f.read()

    if ext in text_files:
        with open(file,'r', encoding='utf-8') as f:
            data = f.read()
        # print(f"Successfully parsed {os.path.basename(file)}")
        return clean_file(data)

SUPPORTED_FORMATS = {
    # structured text — split on headers/structure
    '.md':    'markdown',
    '.html':  'html',
    '.htm':   'html',
    '.xml':   'html',
    # code
    '.py':    'python',
    '.ipynb': 'notebook',
    # config/data
    '.json': 'json',
    '.yaml':  'json',
    '.yml':   'json',
    '.toml':  'json',
    # prose — recursive is genuinely the right default
    '.pdf':   'recursive',
    '.doc':   'recursive',
    '.docx':  'recursive',
    '.ppt':   'recursive',
    '.pptx':  'recursive',
    '.txt':   'recursive',
}

def _split_markdown(text, chunk_size, chunk_overlap):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")]).split_text(text)
    for s in splitter:
        s.page_content = clean_file(s.page_content)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_documents(splitter)

def _split_html(text,chunk_size, chunk_overlap):
    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[("h1", "h1"), ("h2", "h2"), ("h3", "h3")]
        ).split_text(text)
    for s in splitter:
        s.page_content = clean_file(s.page_content)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_documents(splitter)

def _split_python(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return splitter.split_documents([Document(page_content=text)])

def _split_json(text, chunk_size, chunk_overlap):
    return RecursiveJsonSplitter(max_chunk_size=chunk_size).create_documents(
        texts=[json.loads(text)]
    )

def _split_notebook(text, chunk_size, chunk_overlap):
    nb = json.loads(text)
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"])
    return _split_python(code, chunk_size, chunk_overlap)


def _split_recursive(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents([Document(page_content=text)])

SPLITTERS = {
    'markdown':  _split_markdown,
    'html':      _split_html,
    'python':    _split_python,
    'json':      _split_json,
    'notebook':  _split_notebook,
    'recursive': _split_recursive,
}

def split_document(text:str, extention:str, 
                   chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split parsed text into Documents using the strategy suited to the format."""
    strategy = SUPPORTED_FORMATS.get(extention, "recursive")
    handler = SPLITTERS[strategy]
    return handler(text, chunk_size, chunk_overlap)
