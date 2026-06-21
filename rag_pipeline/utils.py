import os
import re
from tika import tika

# Point tika client to remote server 
tika.TikaClientOnly = True
tika.TikaServerEndpoint = "http://tika:9998"

from tika import parser

def supported_file_types():
       
    return {
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', 
            '.txt', '.md', '.py', '.ipynb', '.json', '.yaml', '.yml', 
            '.toml', '.csv', '.html', '.xml'
            }
    
def get_file_extension(file_name):

    _, ext = os.path.splitext(file_name)

    return ext

import re

def clean_file(text):
    """
    Cleans raw extracted text for RAG pipelines.
    - Removes excessive whitespace
    - Preserves paragraph structure
    """

    text = text.replace('\r\n', '\n').replace('\r', '\n') # normalize Windows/Mac line endings 
    text = re.sub(r'[ \t]+', ' ', text) # remove excessive spaces/tabs
    text = re.sub(r' *\n *', '\n', text)  # clean spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text) # collapse multiple blank lines into two

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

    tika_files = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.html', '.xml'}
    text_files = {'.txt', '.md', '.py', '.ipynb', '.json', '.yaml', '.yml', '.toml', '.csv', }

    if get_file_extension(file) in tika_files:
        context = parser.from_file(file)
        print(f"Successfully parsed {os.path.basename(file)}")
        return clean_file(context['content'].strip())

    if get_file_extension(file) in text_files:
        with open(file,'r', encoding='utf-8') as r:
            data = r.read()
        print(f"Successfully parsed {os.path.basename(file)}")
        return clean_file(data)