from eval import run_batch_evaluation, get_score, save_file, get_results
import time
import json
from src.eval_config import *
from src.eval_models import get_metrics, get_run_config

metrics = get_metrics()
run_config = get_run_config()


def main(rag_results, file_path):
    with open(rag_results, "r") as f:
        rag_result_list = [json.loads(line) for line in f]

    try:
        run_batch_evaluation(rag_result_list, metrics, run_config,
                             file_path)

    except Exception as e:
        print(f"Error: {type(e).__name__}")
        raise

if __name__=="__main__":

    print("Starting evaluation...")
    time.sleep(TIME_SLEEP)
    main(RAG_GENERATION_OUTPUT_PATH, RAG_EVALUATION_OUTPUT_PATH)
    print("Evaluation completed.")      