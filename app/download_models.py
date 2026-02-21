import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import os

# CONFIG
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_3B_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SPINOZA_7B_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def download():
    # Attempt to login if HF_TOKEN is provided (for build secrets)
    hf_token = os.getenv("HF_TOKEN")
    source = "Environment Variable"
    
    if not hf_token and os.path.exists("/run/secrets/HF_TOKEN"):
        try:
            with open("/run/secrets/HF_TOKEN", "r") as f:
                hf_token = f.read().strip()
                source = "Build Secret (/run/secrets/HF_TOKEN)"
        except Exception as e:
            print(f"Error reading secret file: {e}")
    
    if hf_token:
        # Masked token for debugging
        masked = hf_token[:4] + "..." + hf_token[-4:] if len(hf_token) > 8 else "****"
        print(f"Attempting login using token from {source} (Token: {masked})")
        try:
            login(token=hf_token)
            print("Successfully logged into Hugging Face Hub.")
        except Exception as e:
            print(f"❌ Login failed: {e}")
            print("Proceeding without authentication (this might fail for gated models)...")
    else:
        print("Warning: No HF_TOKEN found in environment or secrets.")

    print(f"Downloading {BERT_MODEL_NAME}...")
    SentenceTransformer(BERT_MODEL_NAME)
    
    print(f"Downloading {LLAMA_3B_NAME}...")
    AutoTokenizer.from_pretrained(LLAMA_3B_NAME, token=hf_token)
    AutoModelForCausalLM.from_pretrained(LLAMA_3B_NAME, token=hf_token)
    
    print(f"Downloading {SPINOZA_7B_NAME}...")
    AutoTokenizer.from_pretrained(SPINOZA_7B_NAME, token=hf_token)
    AutoModelForCausalLM.from_pretrained(SPINOZA_7B_NAME, token=hf_token)
    
    print("✅ All models downloaded and cached!")

if __name__ == "__main__":
    download()
