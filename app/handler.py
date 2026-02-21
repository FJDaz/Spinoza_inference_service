import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import os
import requests
from loguru import logger

# --- CONFIGURATION ---
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_3B_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SPINOZA_7B_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Remote Inference Config
HYBRID_REMOTE = os.getenv("HYBRID_REMOTE_GENERATION", "false").lower() == "true"
REMOTE_URL = os.getenv("REMOTE_ENGINE_URL")

# Global objects for caching
CLASSIFIER = None
MODELS = {
    "3b": {"model": None, "tokenizer": None},
    "7b": {"model": None, "tokenizer": None}
}
LOCAL_AVAILABLE = False

def load_models():
    """
    Attempt to load models locally. If CUDA is not available or an error occurs,
    mark LOCAL_AVAILABLE as False and proceed (allowing remote fallback).
    """
    global CLASSIFIER, MODELS, LOCAL_AVAILABLE
    
    if not torch.cuda.is_available():
        logger.warning("No CUDA detected. Local inference disabled, switching to Remote Fallback mode.")
        LOCAL_AVAILABLE = False
        return

    try:
        # 1. Load BERT (Vigilance)
        if CLASSIFIER is None:
            logger.info("Loading BERT Vigilance...")
            CLASSIFIER = pipeline("text-classification", model=BERT_MODEL_NAME, device=0)
        
        # Common config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # 2. Load 3B (Sovereign Core)
        if MODELS["3b"]["model"] is None:
            logger.info("Loading Llama 3B...")
            MODELS["3b"]["tokenizer"] = AutoTokenizer.from_pretrained(LLAMA_3B_NAME)
            MODELS["3b"]["model"] = AutoModelForCausalLM.from_pretrained(
                LLAMA_3B_NAME, 
                quantization_config=bnb_config, 
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        
        # Clear cache before loading the next big model
        torch.cuda.empty_cache()
        
        # 3. Load 7B (Expert)
        if MODELS["7b"]["model"] is None:
            logger.info("Loading Spinoza 7B...")
            MODELS["7b"]["tokenizer"] = AutoTokenizer.from_pretrained(SPINOZA_7B_NAME)
            MODELS["7b"]["model"] = AutoModelForCausalLM.from_pretrained(
                SPINOZA_7B_NAME, 
                quantization_config=bnb_config, 
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        
        LOCAL_AVAILABLE = True
        logger.success("All local models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading local models: {e}. Falling back to remote if configured.")
        LOCAL_AVAILABLE = False
        # Final attempt to free memory if load failed
        torch.cuda.empty_cache()

def remote_inference(message, persona, intent):
    """Fallback function to call a remote API (e.g., RunPod)"""
    if not REMOTE_URL:
        return {"error": "No REMOTE_ENGINE_URL configured for fallback"}
    
    logger.info(f"Routing request to remote engine: {REMOTE_URL}")
    try:
        # Assuming RunPod or a similar FastAPI endpoint structure
        payload = {
            "input": {
                "message": message,
                "persona": persona,
                "intent": intent
            }
        }
        # RunPod standard endpoint is /run sync or /runsync
        endpoint = f"{REMOTE_URL}/runsync" if "runpod.net" in REMOTE_URL else REMOTE_URL
        
        # Adding some headers for RunPod auth if a token is in the URL or separate
        # (Usually the token is in the URL for RunPod Serverless or passed via header)
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        # Handle both RunPod format and simple FastAPI format
        content = data.get("output", {}).get("content", data.get("content", "No content in response"))
        
        return {
            "success": True,
            "content": content,
            "model_used": "remote_fallback",
            "intent_detected": intent
        }
    except Exception as e:
        logger.error(f"Remote inference failed: {e}")
        return {"error": f"Hybrid fallback failed: {str(e)}"}

def handler(event):
    """Core Hybrid Inference Handler"""
    input_data = event.get("input", {})
    message = input_data.get("message", "")
    persona = input_data.get("persona", "Spinoza")
    
    if not message:
        return {"error": "No message provided"}

    # --- STEP 1: INTENT DETECTION ---
    # We use a default if BERT isn't loaded locally
    intent = "neutre"
    if CLASSIFIER:
        try:
            analysis = CLASSIFIER(message)[0]
            label = analysis["label"].upper()
            if label == "PERFORMANCE_LEAP": intent = "accord"
            elif label in ["OPPORTUNITY", "NOISE"]: intent = "confusion"
            elif label == "PRICE_ALERT": intent = "resistance"
        except Exception as e:
            logger.warning(f"Intent detection failed: {e}")

    # --- STEP 2: LOCAL OR REMOTE INFERENCE ---
    if LOCAL_AVAILABLE:
        try:
            selected_stack = "7b" if intent == "resistance" else "3b"
            logger.info(f"Local inference using {selected_stack}...")
            
            model_data = MODELS[selected_stack]
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]

            prompt = f"System: Tu es {persona}. Intent: {intent}\nUser: {message}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=250, temperature=0.7)
            response = tokenizer.decode(output[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

            return {
                "success": True,
                "content": response,
                "model_used": f"sovereign_{selected_stack}",
                "intent_detected": intent
            }
        except Exception as e:
            logger.error(f"Local inference crashed: {e}")
            # Fall through to remote if local fails mid-way

    # --- STEP 3: REMOTE FALLBACK ---
    if HYBRID_REMOTE or not LOCAL_AVAILABLE:
        return remote_inference(message, persona, intent)

    return {"error": "No inference engine available (Local disabled and Remote not configured)"}

