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

def clean_file(file):
    """
    Cleans raw extracted text by normalizing excessive whitespace.

    Replaces multiple newline and whitespace patterns with
    standardized line breaks for consistent downstream processing.

    Args:
        file (str): Raw extracted text.

    Returns:
        str: Cleaned text.
    """
    return re.sub('\n\s+\n+','\n\n',file)

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
        return data