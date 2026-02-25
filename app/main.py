import os
import sys
import threading

# Force environment variables before any heavy imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from loguru import logger
logger.info("PYTHON SCRIPT STARTING...")

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import runpod

# Global flag to track model loading status
MODELS_LOADED = False

# Import the core logic from your existing handler
from handler import load_models, handler as core_handler

# --- ENVIRONMENT SWITCH ---
IS_HF = os.getenv("SPACE_ID") is not None
DEFAULT_ENV = "HF_SPACES" if IS_HF else "RUNPOD"
DEPLOY_ENV = os.getenv("DEPLOY_ENVIRONMENT", DEFAULT_ENV).upper()

logger.info(f"Detected Environment: {DEPLOY_ENV}")

# --- HUGGING FACE SPACES App (FastAPI) ---
if DEPLOY_ENV == "HF_SPACES":
    logger.info("Starting in Hugging Face Spaces mode (FastAPI server).")
    
    app = FastAPI()

    @app.get("/")
    async def root():
        """Root endpoint with loading status."""
        return {
            "status": "online",
            "models_loaded": MODELS_LOADED,
            "message": "Spinoza Inference Container is ready." if MODELS_LOADED else "Spinoza Inference Container is starting up and loading models... please wait."
        }

    class HandlerInput(BaseModel):
        """Pydantic model for the input to the handler"""
        message: str
        persona: str = "Spinoza"

    def background_load_models():
        """Function to load models in a separate thread."""
        global MODELS_LOADED
        try:
            logger.info("Background: Starting model load...")
            load_models()
            MODELS_LOADED = True
            logger.info("Background: Models loaded successfully.")
        except Exception as e:
            logger.error(f"Background: Error loading models: {e}")

    @app.on_event("startup")
    async def startup_event():
        """Launch model loading in background to avoid blocking HF health checks."""
        logger.info("Server startup: launching background loading task...")
        threading.Thread(target=background_load_models, daemon=True).start()

    @app.post("/inference")
    async def inference(handler_input: HandlerInput):
        """Inference endpoint with check for model availability."""
        if not MODELS_LOADED:
            return {
                "success": False,
                "error": "Models are still loading. Please try again in 1-2 minutes.",
                "status": "loading"
            }
        
        # Format the input to match the event structure expected by the core handler
        event = {"input": handler_input.dict()}
        result = core_handler(event)
        return result

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)

# --- RUNPOD SERVERLESS ---
elif DEPLOY_ENV == "RUNPOD":
    logger.info("Starting in RunPod Serverless mode.")
    load_models()
    runpod.serverless.start({"handler": core_handler})

else:
    logger.error(f"Unknown DEPLOY_ENVIRONMENT: {DEPLOY_ENV}")
