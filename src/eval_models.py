from functools import lru_cache
from dotenv import load_dotenv
from ragas import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from src.eval_config import *
from ragas.metrics import (
                            LLMContextPrecisionWithReference,
                            LLMContextRecall,
                            Faithfulness,
                            ResponseRelevancy,
                            )
load_dotenv()

@lru_cache(maxsize=1)
def get_generator_llm():
    """
    Initializes and returns a cached instance of the generator LLM.

    Returns:
        LangchainLLMWrapper: Configured LLM instance.

    """
    return LangchainLLMWrapper(
                ChatGroq(
                    model=LLM_GENERATOR,
                    temperature=TEMPERATURE,
                    model_kwargs={"response_format": {"type": "json_object"}},
                    )
                )

@lru_cache(maxsize=1)
def get_evaluator_llm():
    """
    Initializes and returns a cached instance of the evaluator LLM.

    Returns:
        LangchainLLMWrapper: Configured LLM instance.
    """
    return LangchainLLMWrapper(
                ChatGoogleGenerativeAI(
                    model=LLM_JUDGE,
                    temperature=TEMPERATURE,
                    )
                )

@lru_cache(maxsize=1)
def get_eval_embeddings():
    """
    Initializes and returns a cached instance of the evaluator embeddings.

    Returns:
        LangchainEmbeddingsWrapper: Configured embeddings instance.
    """
    return LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
                )
            )

def get_metrics():
    """
    Initializes and returns a list of evaluation metrics.

    Returns:
        list: List of metric instances.
    """
    return [
            LLMContextPrecisionWithReference(llm=get_evaluator_llm()),
            LLMContextRecall(llm=get_evaluator_llm()),
            Faithfulness(llm=get_evaluator_llm()),
            ResponseRelevancy(llm=get_evaluator_llm(), embeddings=get_eval_embeddings()),
            ]

def get_run_config():
    """
    Initializes and returns a RunConfig instance for evaluation.

    Returns:
        RunConfig: Configuration for execution settings.
    """
    return RunConfig(max_workers=MAX_WORKERS, timeout=TIMEOUT)