# imports for synthetic QA Dataset Generation
from collections import defaultdict
from src.eval_models import *
from src.eval_config import TEST_DATA_PATH, CHUNK_SIZE, OVERLAP
import random
from src.eval import (prepare_testset_documents, 
                        generate_qa_dataset, 
                        save_checkpoints, 
                        load_processed_inputs
                        )

generator_llm = get_generator_llm()
generator_embeddings = get_eval_embeddings()
run_config = get_run_config()


def create_doc_set():

    print("Preparing testset documents...")
    docs = prepare_testset_documents(TEST_DATA_PATH, CHUNK_SIZE, OVERLAP)

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

    processed_files = load_processed_inputs(file_path, "filename")

    for file, chunks in docs.items():
        if file in processed_files:
            print(f"Testset for {file} already exists. Skipping...\n")
            continue
        print(len(processed_files), "of", len(docs), "files processed.\n")
        testset_size = define_testset_size(chunks)
        testset = generate_qa_dataset(
                                chunks, generator_llm, generator_embeddings,  
                                run_config=run_config,
                                test_size=testset_size
                            )
        df = testset.to_pandas()
        df["filename"] = file
        save_checkpoints(df, file_path)
        print(f"Saved testset for {file}\n")

if __name__ == "__main__":
    print("Generating test set...")
    docs = create_doc_set()
    create_test_set(docs, "evaluation/testset/new_testset.jsonl")