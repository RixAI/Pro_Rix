# C:\Rix_Dev\Pro_Rix\cloud_services\rix_manager_service\main.py
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal # Ensure all are imported
import uvicorn
import os

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models for Request and Response ---
class ManagerRequest(BaseModel):
    session_id: str
    user_input: Optional[str] = None
    classification: Optional[str] = None # From classifier_node/service
    thinker_output: Optional[str] = None # e.g., direct_answer or plan_summary from thinker_node/service
    tool_result_summary: Optional[str] = None # If a tool was run and summarized
    current_error_message: Optional[str] = None # Any errors accumulated in the graph
    # You might add more context fields like a snippet of agent_history if needed

class ManagerResponse(BaseModel):
    formatted_final_response: str

# --- FastAPI Application ---
app = FastAPI(
    title="Rix Manager Service (Finalizer)",
    description="Receives processed context and formats the final user-facing response. Placeholder V0.1.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Rix Manager Service (Finalizer) starting up... Placeholder logic active.")
    # In the future, this service will initialize:
    # - Vertex AI client (for MANAGER_MODEL)
    # - Load its soul prompt (rix_soul_manager.json)

@app.get("/health", tags=["Health"])
async def health_check():
    logger.info("Health check endpoint called for Manager Service.")
    return {"status": "healthy", "service": "Rix Manager Service V0.1"}

@app.post("/format_response", response_model=ManagerResponse, tags=["ResponseFormatting"])
async def format_final_response(payload: ManagerRequest):
    logger.info(f"Manager service received request for session: {payload.session_id}")
    logger.info(f"Received context - UserInput: '{str(payload.user_input)[:50]}...', Class: {payload.classification}, ThinkerOut: '{str(payload.thinker_output)[:50]}...', ToolRes: '{str(payload.tool_result_summary)[:50]}...', Error: {payload.current_error_message}")

    # --- Placeholder Logic (V0.1) ---
    # This logic will be replaced by an LLM call to the MANAGER_MODEL
    # using rix_soul_manager.json to craft a good response.

    final_text = "I have processed your request (Placeholder response from Rix Manager Service)."

    if payload.current_error_message:
        final_text = f"I encountered an issue: {payload.current_error_message}. Please try rephrasing or ask something else."
    elif payload.thinker_output: # If thinker provided a direct answer or summary
        final_text = str(payload.thinker_output)
    elif payload.tool_result_summary:
        final_text = f"The task resulted in: {payload.tool_result_summary}"
    elif payload.classification == "CHAT":
        final_text = f"Thanks for chatting about '{str(payload.user_input)[:30]}...'! How else can I help? (Manager Service)"
    elif payload.user_input:
         final_text = f"I've processed your input regarding '{str(payload.user_input)[:30]}...' (Manager Service)"


    logger.info(f"Manager service determined final response: {final_text[:100]}")
    return ManagerResponse(formatted_final_response=final_text)

if __name__ == "__main__":
    default_port = int(os.environ.get("PORT", 8085)) # Use a unique port for local testing
    logger.info(f"Attempting to start Manager Uvicorn locally on http://0.0.0.0:{default_port}")
    uvicorn.run("main:app", host="0.0.0.0", port=default_port, reload=True)
