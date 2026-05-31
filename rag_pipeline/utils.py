import os
import re
from tika import tika

# Point tika client to remote server 
tika.TikaClientOnly = True
tika.TikaServerEndpoint = "http://tika:9998"

from tika import parser

def parse(file):

    tika_files = ['.txt','.pdf','.doc','.docx','.xlsx','.ppt','.csv','.html','.xls']
    python_files = ['.py','.ipynb']   

    if file.endswith(tuple(tika_files)):
        context = parser.from_file(file)
        print(f"Successfully parsed {os.path.basename(file)}")
        return clean_file(context['content'].strip())

    if file.endswith(tuple(python_files)):
        with open(file,'r') as r:
            data = r.read()
        print(f"Successfully parsed {os.path.basename(file)}")
        return data

def get_file(path):
    supported_files = ['.txt','.pdf','.docx','.xlsx','.py','.ipynb','.ppt','.csv','.html','.doc','.xls']
    files_in_dir = os.listdir(path)
    files = []
    for file in files_in_dir:
        if file.endswith(tuple(supported_files)):
            files.append(os.path.join(path,file))
    return files

def clean_file(file):
    return re.sub('\n\s+\n+','\n\n',file)
    
def text_parser(file_path):
    text=""
    n=0
    # for file in files:   
    parsed_text = parse(file_path) 
    if parsed_text:
        text+=parsed_text+"\n\n"
        n+=1
        print(f"Successfully parsed {os.path.basename(file_path)}")
    else:
        print(f"Error parsing {os.path.basename(file_path)}")
    
    return text, n
