import os
import sys

# Force environment variables before any heavy imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from loguru import logger
logger.info("PYTHON SCRIPT STARTING...")

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import runpod

# Import the core logic from your existing handler
from handler import load_models, handler as core_handler

# --- ENVIRONMENT SWITCH ---
# Decide which environment to run based on an environment variable.
# Automatically detect HF Spaces via standard HF variables if not set.
IS_HF = os.getenv("SPACE_ID") is not None
DEFAULT_ENV = "HF_SPACES" if IS_HF else "RUNPOD"
DEPLOY_ENV = os.getenv("DEPLOY_ENVIRONMENT", DEFAULT_ENV).upper()

logger.info(f"Detected Environment: {DEPLOY_ENV}")


# --- HUGGING FACE SPACES App (FastAPI) ---
if DEPLOY_ENV == "HF_SPACES":
    logger.info("Starting in Hugging Face Spaces mode (FastAPI server).")
    
    app = FastAPI()

    class HandlerInput(BaseModel):
        """Pydantic model for the input to the handler"""
        message: str
        persona: str = "Spinoza"

    @app.on_event("startup")
    def startup_event():
        """Load models when the server starts"""
        logger.info("Server starting up...")
        load_models()
        logger.info("Models loaded successfully.")

    @app.post("/inference")
    async def inference(handler_input: HandlerInput):
        """
        Inference endpoint. Takes the same input as the RunPod handler
        but wrapped in a FastAPI request.
        """
        # Format the input to match the event structure expected by the core handler
        event = {"input": handler_input.dict()}
        
        # Call the core handler logic
        result = core_handler(event)
        
        return result

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)

# --- RUNPOD SERVERLESS ---
elif DEPLOY_ENV == "RUNPOD":
    logger.info("Starting in RunPod Serverless mode.")
    
    # Load models immediately for RunPod's serverless environment
    load_models()
    
    # Start the RunPod serverless handler
    runpod.serverless.start({"handler": core_handler})

else:
    logger.error(f"Unknown DEPLOY_ENVIRONMENT: {DEPLOY_ENV}")

