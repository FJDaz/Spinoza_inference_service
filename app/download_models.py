import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import os

# CONFIG
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_3B_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SPINOZA_7B_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def download():
    print(f"Downloading {BERT_MODEL_NAME}...")
    SentenceTransformer(BERT_MODEL_NAME)
    
    # We don't use quantization during download to ensure we get the full weights if needed, 
    # but transformers usually caches the safetensors.
    print(f"Downloading {LLAMA_3B_NAME}...")
    AutoTokenizer.from_pretrained(LLAMA_3B_NAME)
    AutoModelForCausalLM.from_pretrained(LLAMA_3B_NAME)
    
    print(f"Downloading {SPINOZA_7B_NAME}...")
    AutoTokenizer.from_pretrained(SPINOZA_7B_NAME)
    AutoModelForCausalLM.from_pretrained(SPINOZA_7B_NAME)
    
    print("✅ All models downloaded and cached!")

if __name__ == "__main__":
    download()
