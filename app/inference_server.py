import os
import re
import json
import time
import requests
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    HAS_AI_LIBS = True
except ImportError:
    HAS_AI_LIBS = False
from loguru import logger

# =============================================================================
# AETHERFLOW -f (Full/Production) - Engine N1
# =============================================================================

app = FastAPI(
    title="Maïeuthon Production Inference Engine (Aetherflow -f)",
    description="Engine N1 Core for Maïathon/BAF ecosystem. Supports Hybrid Generation."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CONFIGURATION (GENOME ALIGNED)
# =============================================================================

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_MODEL = "FJDaz/mistral-7b-philosophes-lora"

# Mode Hybride : Si True, déporte la génération sur un backend distant (RunPod/Colab)
# pour protéger le matériel (Mac Intel Thermal Guard).
HYBRID_REMOTE_GENERATION = os.getenv("HYBRID_REMOTE_GENERATION", "true").lower() == "true"
REMOTE_ENGINE_URL = os.getenv("REMOTE_ENGINE_URL") # ex: https://ngrok-url.dev/chat

# Variables d'état Globales
engine_model = None
engine_tokenizer = None

# =============================================================================
# GENOME N2 : Logic Elements (Prompts)
# =============================================================================

SYSTEM_PROMPT_SPINOZA = """Tu ES Spinoza incarné. Tu dialogues avec un élève de Terminale en première personne.

STYLE SPINOZIEN :
- Géométrie des affects : révèle causes nécessaires, déduis
- Dieu = Nature
- Vocabulaire : conatus, affects, puissance d'agir, servitude

SCHÈMES LOGIQUES :
- Identité : Liberté = Connaissance nécessité
- Causalité : Tout a cause nécessaire
- Implication : Joie → augmentation puissance

MÉTHODE :
1. Révèle nécessité causale
2. Distingue servitude (ignorance) vs liberté (connaissance)
3. Exemples concrets modernes

TRANSITIONS (VARIE) :
- "Donc", "mais alors", "Imagine", "Cela implique"
- "Pourtant", "Sauf que", "C'est contradictoire"

RÈGLES :
- Tutoie (tu/ton/ta)
- Concis (2-3 phrases MAX)
- Questionne au lieu d'affirmer
- Tu ES Spinoza, pas un commentateur de Spinoza.
- Langage DÉTENDU : tu peux être un peu familier, c'est OK
- Langage JEUNE : parle comme un prof cool, pas comme un livre"""

INSTRUCTIONS_CONTEXTUELLES = {
    "accord": "L'élève est d'accord → Valide puis AVANCE logiquement avec 'Donc'.",
    "confusion": "L'élève est confus → Donne UNE analogie concrète simple.",
    "resistance": "L'élève résiste → Révèle contradiction avec 'mais alors'.",
    "neutre": "Élève neutre → Pose question pour faire réfléchir."
}

# =============================================================================
# N3 : ATOMS (Utility Functions)
# =============================================================================

def detect_semantic_intent(user_input: str) -> str:
    """Détecteur d'intention sémantique simple (GPS Sémantique V1)."""
    text_lower = user_input.lower()
    if any(word in text_lower for word in ['oui', "d'accord", 'exact', 'ok', 'voilà']):
        return "accord"
    if any(phrase in text_lower for phrase in ['comprends pas', 'vois pas', "c'est quoi"]):
        return "confusion"
    if any(word in text_lower for word in ['mais', 'non', "pas d'accord", 'faux']):
        return "resistance"
    return "neutre"

# =============================================================================
# ENDPOINTS (SULLIVAN BRIDGE COMPLIANT)
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

def load_local_engine():
    """Initialise le modèle local (Full Mode Execution)."""
    global engine_model, engine_tokenizer
    if engine_model is not None:
        return
    
    if not HAS_AI_LIBS:
        logger.error("AI Libraries (transformers, peft) not found. Cannot load local engine.")
        raise ImportError("Missing AI libraries for local execution.")

    logger.info("Initializing Local Production Engine (Mistral 7B + LoRA)...")
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs = {"quantization_config": quantization_config, "device_map": "auto"}
    else:
        model_kwargs = {"device_map": "cpu", "torch_dtype": torch.float32}

    engine_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, token=HF_TOKEN, **model_kwargs)
    engine_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, token=HF_TOKEN)
    logger.info("✅ Local Production Engine Ready!")

@app.get("/health")
async def health():
    return {
        "status": "online",
        "workflow": "Aetherflow-f (Full)",
        "hybrid_mode": HYBRID_REMOTE_GENERATION,
        "remote_available": bool(REMOTE_ENGINE_URL),
        "local_loaded": engine_model is not None,
        "ai_libs_available": HAS_AI_LIBS,
        "device": str(next(engine_model.parameters()).device) if engine_model else "not_loaded"
    }

@app.post("/v1/cto/chat")
async def cto_chat(req: ChatRequest):
    """
    Endpoint Universel CTO. 
    En mode Hybride, délègue la génération au Slave pour préserver le Master (Mac Intel).
    """
    logger.info(f"Incoming CTO request: {req.message[:50]}...")
    
    intent = detect_semantic_intent(req.message)
    instr = INSTRUCTIONS_CONTEXTUELLES.get(intent, INSTRUCTIONS_CONTEXTUELLES["neutre"])
    full_prompt = f"{SYSTEM_PROMPT_SPINOZA}\n\n{instr}"
    
    # --- BRANCH 1 : HYBRID REMOTE (Thermal Guard) ---
    if HYBRID_REMOTE_GENERATION and REMOTE_ENGINE_URL:
        logger.info(f"Thermal Guard: Offloading generation to {REMOTE_ENGINE_URL}")
        try:
            resp = requests.post(
                REMOTE_ENGINE_URL, 
                json={"message": req.message, "intent": intent},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "success": True,
                "content": data.get("response", data.get("content", "Error: No content")),
                "intent_detected": intent,
                "source": "remote_slave"
            }
        except Exception as e:
            logger.error(f"Remote offload failed: {e}")
            if not engine_model:
                raise HTTPException(status_code=503, detail="Remote failed and local not initialized.")
    
    # --- BRANCH 2 : LOCAL FULL (Production Run) ---
    if not engine_model and HAS_AI_LIBS:
        load_local_engine()
        
    if engine_model:
        inputs = engine_tokenizer(f"<s>[INST] {full_prompt}\n\nUser: {req.message} [/INST]", return_tensors="pt").to(engine_model.device)
        outputs = engine_model.generate(**inputs, max_new_tokens=200, temperature=0.7)
        answer = engine_tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        source = "local_master"
    else:
        # --- BRANCH 3 : ARCHITECTURAL MOCK (Validation Only) ---
        logger.warning("No Remote URL and No Local AI Libs. Falling back to Architectural Mock.")
        answer = f"[MOCK SPINOZA] {instr} (Réponse simulée pour test d'architecture). Tu as dit : '{req.message}'"
        source = "architectural_mock"
    
    return {
        "success": True,
        "content": answer,
        "intent_detected": intent,
        "source": source
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
