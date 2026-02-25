import os
import threading
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel

# Initialisation immédiate de FastAPI (Univoque)
app = FastAPI(title="Spinoza Inference Service")

MODELS_LOADED = False

# Tentative d import du handler
try:
    from handler import load_models, handler as core_handler
    logger.info("Handler imported successfully")
except ImportError as e:
    logger.error(f"Import error: {e}")
    def load_models(): pass
    def core_handler(x): return {"error": "Handler not found"}

@app.get("/")
async def root():
    return {
        "status": "online",
        "models_loaded": MODELS_LOADED,
        "environment": os.getenv("SPACE_ID", "local"),
        "message": "Welcome to Spinoza Inference Service. Models are loading..." if not MODELS_LOADED else "Models loaded and ready."
    }

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}

class HandlerInput(BaseModel):
    message: str
    persona: str = "Spinoza"

def background_load_models():
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
    # Toujours lancer le chargement, que ce soit HF ou autre
    threading.Thread(target=background_load_models, daemon=True).start()

@app.post("/inference")
async def inference(handler_input: HandlerInput):
    if not MODELS_LOADED:
        return {"success": False, "error": "Models loading", "status": "loading"}
    event = {"input": handler_input.dict()}
    return core_handler(event)

if __name__ == "__main__":
    import uvicorn
    # Le port 7860 est impératif pour HF Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)
