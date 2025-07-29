# preload_models.py

import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Suppress warnings for a cleaner build log.
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Model configurations from our core_logic.py
PRODUCTION_LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def download_embedding_model():
    print(f"--> Downloading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"--> SUCCESS: Embedding model cached.")
    except Exception as e:
        print(f"--> ERROR downloading embedding model: {e}")

def download_llm(model_name):
    print(f"--> Downloading LLM: {model_name}")
    try:
        # These commands download all necessary files without fully loading the model
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        print(f"--> SUCCESS: LLM components cached.")
    except Exception as e:
        print(f"--> ERROR downloading LLM: {model_name}: {e}")

def main():
    print("🚀 Starting model pre-warming...")
    download_embedding_model()
    download_llm(PRODUCTION_LLM_MODEL_NAME)
    print("✅ Model pre-warming finished.")

if __name__ == "__main__":
    main()