import os
import threading
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
import runpod

# Import de la logique du handler
from handler import load_models, handler as core_handler

# Détection de l environnement
DEPLOY_ENV = os.getenv("DEPLOY_ENVIRONMENT", "LOCAL").upper()
if os.getenv("SPACE_ID"):
    DEPLOY_ENV = "HF_SPACES"

logger.info(f"🚀 Mode détecté : {DEPLOY_ENV}")

# --- MODE RUNPOD SERVERLESS (Homeos Mode) ---
if DEPLOY_ENV == "RUNPOD":
    logger.info("📡 Homeos Mode: Starting RunPod Serverless Worker...")
    # BAKE STRATEGY: On charge les modèles en VRAM avant de démarrer le worker
    # Cela garantit que le worker est prêt à répondre dès qu il est actif.
    load_models() 
    runpod.serverless.start({"handler": core_handler})

# --- MODE FASTAPI (Local ou HF Spaces) ---
else:
    logger.info("💻 Web Service Mode: Starting FastAPI...")
    app = FastAPI(title="Spinoza Inference Service")
    MODELS_LOADED = False

    class HandlerInput(BaseModel):
        message: str = None
        prompt: str = None  # Alias pour compatibilité
        persona: str = "Spinoza"

    @app.get("/")
    async def root():
        return {
            "status": "online",
            "models_loaded": MODELS_LOADED,
            "message": "Service opérationnel. Utilisez /inference (POST)."
        }

    def background_load_models():
        global MODELS_LOADED
        try:
            load_models()
            MODELS_LOADED = True
            logger.success("Modèles chargés avec succès.")
        except Exception as e:
            logger.error(f"Erreur chargement modèles : {e}")

    @app.on_event("startup")
    async def startup_event():
        threading.Thread(target=background_load_models, daemon=True).start()

    @app.post("/inference")
    async def inference(handler_input: HandlerInput):
        if not MODELS_LOADED:
            return {"success": False, "error": "Chargement des modèles en cours...", "status": "loading"}
        
        # Unification message/prompt
        text = handler_input.message or handler_input.prompt
        if not text:
            return {"error": "Champ message ou prompt manquant"}
            
        event = {
            "input": {
                "message": text,
                "persona": handler_input.persona
            }
        }
        return core_handler(event)

    if __name__ == "__main__":
        import uvicorn
        port = int(os.getenv("PORT", 7860))
        uvicorn.run(app, host="0.0.0.0", port=port)
