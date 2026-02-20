import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import os
from loguru import logger

# --- CONFIGURATION ---
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLAMA_3B_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SPINOZA_7B_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Base to be updated with LoRA if needed

# Global objects for caching
CLASSIFIER = None
MODELS = {
    "3b": {"model": None, "tokenizer": None},
    "7b": {"model": None, "tokenizer": None}
}

def load_models():
    global CLASSIFIER, MODELS
    
    # 1. Load BERT (Vigilance)
    if CLASSIFIER is None:
        logger.info("Loading BERT Vigilance...")
        CLASSIFIER = pipeline("text-classification", model=BERT_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
    
    # 2. Load 3B (Sovereign Core)
    if MODELS["3b"]["model"] is None:
        logger.info("Loading Llama 3B...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        MODELS["3b"]["tokenizer"] = AutoTokenizer.from_pretrained(LLAMA_3B_NAME)
        MODELS["3b"]["model"] = AutoModelForCausalLM.from_pretrained(
            LLAMA_3B_NAME, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
    
    # 3. Load 7B (Expert)
    if MODELS["7b"]["model"] is None:
        logger.info("Loading Spinoza 7B...")
        # Re-using the same bnb_config is fine
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        MODELS["7b"]["tokenizer"] = AutoTokenizer.from_pretrained(SPINOZA_7B_NAME)
        MODELS["7b"]["model"] = AutoModelForCausalLM.from_pretrained(
            SPINOZA_7B_NAME, 
            quantization_config=bnb_config, 
            device_map="auto"
        )

def handler(event):
    """Core Inference Handler"""
    input_data = event.get("input", {})
    message = input_data.get("message", "")
    persona = input_data.get("persona", "Spinoza").lower()
    
    if not message:
        return {"error": "No message provided"}

    # --- STEP 1: VIGILANCE (BERT) ---
    analysis = CLASSIFIER(message)[0]
    # BERT targets: "PRICE_ALERT", "PERFORMANCE_LEAP", "OPPORTUNITY", "NOISE"
    # We map them to Maïieutique intents
    label = analysis["label"].upper()
    
    intent = "neutre"
    if label == "PERFORMANCE_LEAP":
        intent = "accord"
    elif label == "OPPORTUNITY" or label == "NOISE":
        intent = "confusion"
    elif label == "PRICE_ALERT":
        intent = "resistance"

    # --- STEP 2: INFERENCE ---
    selected_stack = "3b"
    if intent == "resistance":
        selected_stack = "7b"
        logger.info("Elevation to 7B triggered by resistance.")

    model_data = MODELS[selected_stack]
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]

    # Prompt construction
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
