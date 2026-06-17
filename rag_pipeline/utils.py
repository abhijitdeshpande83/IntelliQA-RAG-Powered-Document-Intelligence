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
    return re.sub('\n\s+\n+','\n\n',file)

def parse(file):

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