# imports for synthetic QA Dataset Generation
from ragas import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama                     # ollama can run locally
from langchain_huggingface import HuggingFaceEmbeddings
from src.eval import prepare_testset_documents, generate_qa_dataset, save_checkpoints
from collections import defaultdict
import random

generator_llm = LangchainLLMWrapper(
                ChatOllama(
                    model="qwen2.5",
                    temperature=0,
                    # model_kwargs={"response_format":{"type":"json_object"}}
                    format="json"
                    )
                )

generator_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(
                    model_name="BAAI/bge-large-en-v1.5",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
                    )
                )    

run_config = RunConfig(max_workers=2, timeout=180)   


def create_doc_set():

    print("Preparing testset documents...")
    docs = prepare_testset_documents("eval_data", chunk_size=1000, chunk_overlap=150)

    group_docs = defaultdict(list)
    
    for doc in docs:
        file_name = doc.metadata['filename']
        content = doc.page_content
        
        if len(content) < 700:
            continue
        if len(content.split()) <= 100:
                continue
        
        group_docs[file_name].append(doc)

    sampled_docs = {
                file: random.sample(chunks, min(len(chunks), 50)) 
                    for file, chunks in group_docs.items()
                    }
    print(f"Prepared {len(sampled_docs)} documents for testset generation.")
    return sampled_docs

def define_testset_size(chunks):
    if len(chunks) > 40:
        return 3
    elif len(chunks) > 20:
        return 2
    else:
        return 1

def create_test_set(docs, file_path):
    for file, chunks in docs.items():
        testset_size = define_testset_size(chunks)
        testset = generate_qa_dataset(
                                chunks, generator_llm, generator_embeddings,  
                                run_config=run_config,
                                test_size=testset_size
                            )
        df = testset.to_pandas()
        df["filename"] = file
        save_checkpoints(df, file_path)
        print(f"Saved testset for {file}")

if __name__ == "__main__":
    print("Generating test set...")
    docs = create_doc_set()
    create_test_set(docs, "test_data/new_testset.jsonl")