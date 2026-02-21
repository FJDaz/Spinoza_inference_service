import os
# Force OMP_NUM_THREADS to 1 before any other imports to fix libgomp error
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from huggingface_hub import login, snapshot_download

# CONFIG
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_3B_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SPINOZA_7B_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def download():
    # Attempt to login if token is provided
    hf_token = os.getenv("inference_space") or os.getenv("HF_TOKEN")
    
    if not hf_token:
        secret_path = "/run/secrets/inference_space"
        if os.path.exists(secret_path):
            try:
                with open(secret_path, "r") as f:
                    hf_token = f.read().strip()
            except Exception as e:
                print(f"Error reading secret file: {e}")
    
    if hf_token:
        print("Logging into Hugging Face Hub...")
        try:
            login(token=hf_token)
        except Exception as e:
            print(f"❌ Login failed: {e}")

    # Use snapshot_download to only download files to cache without loading into RAM
    print(f"Downloading {BERT_MODEL_NAME}...")
    snapshot_download(repo_id=BERT_MODEL_NAME)
    
    print(f"Downloading {LLAMA_3B_NAME}...")
    snapshot_download(repo_id=LLAMA_3B_NAME, token=hf_token)
    
    print(f"Downloading {SPINOZA_7B_NAME}...")
    snapshot_download(repo_id=SPINOZA_7B_NAME, token=hf_token)
    
    print("✅ All models downloaded to cache!")

if __name__ == "__main__":
    download()
