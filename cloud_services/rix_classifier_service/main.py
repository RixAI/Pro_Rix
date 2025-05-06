# C:\Rix_Dev\Pro_Rix\cloud_services\rix_classifier_service\main.py
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os

# --- ADD THESE TYPING IMPORTS ---
from typing import List, Optional, Dict, Literal # Literal was there, ensure List & Optional for Pydantic

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models for Request and Response ---
class ClassifierRequest(BaseModel):
    session_id: str = Field(..., example="cli_session_XYZ_turn1")
    user_input: str = Field(..., example="Hello Rix, can you help me create a file?")
    # Optional: history_context: List[Dict] = Field(default_factory=list)

class ClassifierResponse(BaseModel):
    classification: Literal["CHAT", "ASK", "WORK", "RCD", "UNKNOWN"]
    confidence: Optional[float] = Field(default=None)
    entities: Optional[List[Dict]] = Field(default_factory=list)
    details: str
    received_input: str

# --- FastAPI Application ---
app = FastAPI(
    title="Rix Classifier Service",
    description="Receives user input and provides a classification. Placeholder V0.2 (No ADK Agent call yet).",
    version="0.2.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Rix Classifier Service starting up...")
    # If you were using an ADK agent, you would initialize it here:
    # global adk_agent_instance
    # from google.adk.agents import Agent # Example
    # adk_agent_instance = Agent(name="simple_classifier_agent", model="gemini-1.5-flash-preview-0514", instruction="Classify this.")
    # logger.info("ADK Agent instance created (placeholder).")
    logger.info("Service ready to receive requests (placeholder logic).")

@app.get("/health", tags=["Health"])
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "healthy", "service": "Rix Classifier Service V0.2"}

@app.post("/classify", response_model=ClassifierResponse, tags=["Classification"])
async def classify_input_endpoint(request: Request, payload: ClassifierRequest): # Renamed function slightly
    """
    Receives user input and returns a classification.
    Currently returns a placeholder/dummy classification.
    NO ADK AGENT CALL YET to avoid AttributeError.
    """
    # id_token = request.headers.get("Authorization") # For debugging auth later
    # logger.info(f"Authorization header: {id_token[:30] if id_token else 'None'}")
    logger.info(f"Received classification request for session_id: {payload.session_id}")
    logger.info(f"User input: '{payload.user_input[:100]}...'")

    # --- Placeholder Logic (V0.2 - No ADK Agent call) ---
    classification_output: Literal["CHAT", "ASK", "WORK", "RCD", "UNKNOWN"] = "UNKNOWN"
    confidence_output: Optional[float] = 0.5
    entities_output: List[Dict] = []
    details_output = "Placeholder classification from service v0.2"

    user_input_lower = payload.user_input.lower()
    if "file" in user_input_lower or "create" in user_input_lower or "run" in user_input_lower or "execute" in user_input_lower:
        classification_output = "WORK"; confidence_output = 0.8
        if "file" in user_input_lower: entities_output.append({"type": "OBJECT", "value": "file"})
    elif "what can you do" in user_input_lower or "help" in user_input_lower or "who are you" in user_input_lower:
        classification_output = "ASK"; confidence_output = 0.75
    elif "hello" in user_input_lower or "hi" in user_input_lower or "how are you" in user_input_lower:
        classification_output = "CHAT"; confidence_output = 0.85
    else:
        classification_output = "CHAT"; confidence_output = 0.6 # Default to CHAT for now

    logger.info(f"Placeholder classification determined: {classification_output}")

    return ClassifierResponse(
        classification=classification_output,
        confidence=confidence_output,
        entities=entities_output,
        details=details_output,
        received_input=payload.user_input
    )

# This part is for running locally with `python main.py` for testing,
# Cloud Run uses the CMD from Dockerfile.
if __name__ == "__main__":
    default_port = int(os.environ.get("PORT", 8080))
    logger.info(f"Attempting to start Uvicorn locally on http://0.0.0.0:{default_port}")
    uvicorn.run("main:app", host="0.0.0.0", port=default_port, reload=True)
