from eval import run_batch_evaluation, get_score, save_file, get_results
import time
import argparse
import json
from ragas import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.metrics import (
                            LLMContextPrecisionWithReference,
                            LLMContextRecall,
                            Faithfulness,
                            ResponseRelevancy,
                        )

evaluator_llm = LangchainLLMWrapper(
                    ChatGoogleGenerativeAI(
                        model="gemini-flash-lite-latest",
                        temperature=0,
                        # model_kwargs={"response_format": {"type": "json_object"}},
                    )
                )
evaluator_embeddings = LangchainEmbeddingsWrapper(
                        HuggingFaceEmbeddings(
                            model_name="BAAI/bge-large-en-v1.5",
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
                            )
                        )

metrics = [
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    LLMContextRecall(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
]

run_config = RunConfig(max_workers=1, timeout=600)


def main(rag_results, file_path):
    with open(rag_results, "r") as f:
        rag_result_list = [json.loads(line) for line in f]

    try:
        run_batch_evaluation(rag_result_list, metrics, run_config, file_path)

    except Exception as e:
        print(type(e))
        print(e)
        raise

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=int, default=0)
    parser.add_argument("--rag_results", required=True)
    parser.add_argument("--file_path", required=True)

    args = parser.parse_args()

    print("Starting evaluation...")
    time.sleep(args.sleep)
    main(rag_results=args.rag_results, file_path=args.file_path)
    print("Evaluation completed.")      