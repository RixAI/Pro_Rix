# C:\Rix_Dev\Pro_Rix\cloud_services\rix_classifier_service\main.py
import logging
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel, Field
import uvicorn # For local testing if needed
import os # To get PORT from environment for Cloud Run
from typing import Literal # This is absolutely essential.
from typing import List, Optional, Dict, Literal # Literal was already there, ensure List and Optional are too
# Basic logging configuration. Cloud Run will capture stdout/stderr.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models for Request and Response ---
class ClassifierRequest(BaseModel):
    session_id: str = Field(..., example="cli_session_XYZ_turn1")
    user_input: str = Field(..., example="Hello Rix, can you help me create a file?")
    # Optional: history_context: List[Dict] = Field(default_factory=list)

class ClassifierResponse(BaseModel):
    classification: Literal["CHAT", "ASK", "WORK", "RCD", "UNKNOWN"] = Field(..., example="WORK")
    confidence: Optional[float] = Field(default=None, example=0.95)
    entities: Optional[List[Dict]] = Field(default_factory=list, example=[{"type": "TASK", "value": "create a file"}])
    details: str = Field(default="Placeholder response from rix-classifier-service v0.1")
    received_input: str # To echo back for verification

# --- FastAPI Application ---
app = FastAPI(
    title="Rix Classifier Service",
    description="Receives user input and provides a classification. Placeholder V0.1.",
    version="0.1.0"
)

# --- Health Check Endpoint ---
@app.get("/health", tags=["Health"])
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "healthy", "service": "Rix Classifier Service V0.1"}

# --- Classifier Endpoint ---
@app.post("/classify", response_model=ClassifierResponse, tags=["Classification"])
async def classify_input(request: Request, payload: ClassifierRequest):
    """
    Receives user input and returns a classification.
    Currently returns a placeholder/dummy classification.
    """
    # Log incoming request headers for OIDC token debugging (optional, remove in prod)
    # logger.info(f"Request headers: {request.headers}")
    # id_token = request.headers.get("Authorization", "No Authorization Header")
    # logger.info(f"Authorization Header: {id_token[:30] if id_token != 'No Authorization Header' else id_token}...")

    logger.info(f"Received classification request for session_id: {payload.session_id}")
    logger.info(f"User input: '{payload.user_input[:100]}...'")

    # --- Placeholder Logic (V0.1) ---
    # Replace this with actual LLM call using invoke_classifier_agent from vertex_llm.py later
    classification_output = "UNKNOWN"
    confidence_output = 0.5
    entities_output = []

    user_input_lower = payload.user_input.lower()
    if "file" in user_input_lower or "create" in user_input_lower or "run" in user_input_lower or "execute" in user_input_lower:
        classification_output = "WORK"
        confidence_output = 0.8
        if "file" in user_input_lower: entities_output.append({"type": "OBJECT", "value": "file"})
    elif "what can you do" in user_input_lower or "help" in user_input_lower or "who are you" in user_input_lower:
        classification_output = "ASK"
        confidence_output = 0.75
    elif "hello" in user_input_lower or "hi" in user_input_lower or "how are you" in user_input_lower:
        classification_output = "CHAT"
        confidence_output = 0.85
    else:
        # Default fallback if no keywords match
        classification_output = "CHAT" # Or "UNKNOWN"
        confidence_output = 0.6

    logger.info(f"Placeholder classification determined: {classification_output}")

    return ClassifierResponse(
        classification=classification_output, # type: ignore
        confidence=confidence_output,
        entities=entities_output,
        details=f"Placeholder classification for '{payload.user_input[:30]}...'",
        received_input=payload.user_input
    )

# --- Main entry point for local testing (optional) ---
if __name__ == "__main__":
    # This is for running locally with uvicorn, e.g., `python main.py`
    # Cloud Run uses a different mechanism (gunicorn with uvicorn workers).
    port = int(os.environ.get("PORT", 8080)) # Default to 8080 if PORT not set
    logger.info(f"Starting Uvicorn locally on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# --- Add Literal import for Pydantic model ---
from typing import Literal # Crucial for ClassifierResponse
