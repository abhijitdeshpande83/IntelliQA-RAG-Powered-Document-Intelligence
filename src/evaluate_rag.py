from eval import run_batch_evaluation, get_score, save_file, get_results
import time
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


def main():
    with open("test_data/rag_results.jsonl", "r") as f:
        rag_results = [json.loads(line) for line in f]

    file_path = "evaluation/rag_eval_v8.jsonl"
    try:
        run_batch_evaluation(rag_results, metrics, run_config, file_path)

    except Exception as e:
        print(type(e))
        print(e)
        raise

if __name__=="__main__":
    print("Starting evaluation...")
    # time.sleep(9000)
    main()