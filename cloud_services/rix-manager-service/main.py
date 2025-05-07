# --- START OF FILE rix-manager-service/main.py ---
# Version: V60.1 (Intelligent Entry Point & Dispatcher)

import sys
import os
from pathlib import Path
import logging
import json
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Body, Request as FastAPIRequest
from pydantic import BaseModel, Field

# --- 1. Rix Brain Path Setup & Core Imports ---
# This section assumes a specific directory structure when deployed:
# /app/ (Cloud Run copies your files here)
#   main.py (this file)
#   config.json
#   Rix_Brain/
#     core/
#     services/
#     agents/
#     ...etc.

# Ensure PROJECT_ROOT is /app when running in Cloud Run
# For local testing, it would be your actual project root (e.g., C:\Rix\Pro_Rix)
# Path(__file__).resolve().parent is the directory of main.py
# Path(__file__).resolve().parent.parent is the directory containing main.py's dir

APP_ROOT_DIR = Path(__file__).resolve().parent # This is /app in Cloud Run
# If Rix_Brain is directly inside /app, then it's APP_ROOT_DIR / "Rix_Brain"
# If config.json is directly inside /app, it's APP_ROOT_DIR / "config.json"

# Adjust if your deployment structure for Rix_Brain and config.json is different
# relative to where main.py for the service lands.
# This structure assumes Rix_Brain and config.json are at the same level as the service's main.py
# when deployed, or one level up if main.py is in an 'app' subdir of the service.
# For simplicity, let's assume for Cloud Run, you package Rix_Brain and config.json
# at the root of your deployment package, so they are accessible like this:
PROJECT_ROOT_FOR_IMPORTS = Path(".") # Represents the root of the deployed package in Cloud Run

RIX_BRAIN_PATH = PROJECT_ROOT_FOR_IMPORTS / "Rix_Brain"
CONFIG_JSON_PATH = PROJECT_ROOT_FOR_IMPORTS / "config.json" # Ensure config.json is here

if str(PROJECT_ROOT_FOR_IMPORTS.resolve()) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS.resolve()))
if str(RIX_BRAIN_PATH.resolve()) not in sys.path: # Ensure Rix_Brain itself is importable
    sys.path.insert(0, str(RIX_BRAIN_PATH.resolve()))

logger = logging.getLogger(__name__) # Get logger after path setup potentially modifies logging config via imports
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

try:
    from Rix_Brain.core import initialization
    from Rix_Brain.core import config_manager as rix_config
    from Rix_Brain.core import utils as rix_utils
    from Rix_Brain.services import vertex_llm # Requires invoke_llm_for_json_action to be added
    from Rix_Brain.core import global_state as RixState
    from Rix_Brain.core.exceptions import LLMInvocationError
    logger.info("Successfully imported Rix_Brain modules for Manager Service.")
except ImportError as e:
    logger.error(f"FATAL: Failed to import Rix_Brain modules in Manager Service: {e}", exc_info=True)
    # This is a critical failure; the service likely can't run.
    # Cloud Run might not start, or requests will fail.
    initialization = None # Make linters happy, but service is broken

# --- 2. FastAPI App Setup ---
app = FastAPI(
    title="Rix Manager Service",
    description="V60.1 - Intelligent User Interface ('Rix' Persona) and Task Dispatcher.",
    version="60.1.0"
)

# --- 3. Global State for Initialization & Soul Prompt ---
RIX_CORE_INITIALIZED = False
MANAGER_SOUL_PROMPT_TEMPLATE: Optional[str] = None
SERVICE_STARTUP_ERROR: Optional[str] = None

# --- 4. Pydantic Models for API ---
class ManagerTaskRequest(BaseModel):
    session_id: str
    current_processing_stage: str = Field(description="e.g., 'INITIAL_USER_INPUT', 'FINALIZE_THINKER_REPORT'")
    user_input: Optional[str] = None
    history_context: List[Dict[str, Any]] = Field(default_factory=list)
    thinker_report_content: Optional[Any] = None # str or dict
    classifier_output: Optional[Dict[str, Any]] = None
    original_user_input: Optional[str] = None # For when handling classifier result

class ManagerTaskResponse(BaseModel):
    action: str
    # Other fields will be dynamic based on the action, Pydantic handles this with Extra.allow
    # For stricter validation, define specific models for each action type if needed later.
    class Config:
        extra = "allow" # Allow additional fields not explicitly defined

# --- 5. Startup Event for Initialization ---
@app.on_event("startup")
async def startup_event():
    global RIX_CORE_INITIALIZED, MANAGER_SOUL_PROMPT_TEMPLATE, SERVICE_STARTUP_ERROR
    
    # Override config path for config_manager if needed for Cloud Run
    # config_manager.CONFIG_FILE_PATH = CONFIG_JSON_PATH # If config_manager doesn't find it by default
    # Ensure Rix_Auth path is also correctly discoverable or not strictly needed if using runtime SA
    # config_manager.SERVICE_ACCOUNT_KEY_PATH = APP_ROOT_DIR / "Rix_Brain" / "Rix_Auth" / "yourkey.json" # Avoid this on Cloud Run

    if not RIX_CORE_INITIALIZED:
        logger.info("Manager Service Cold Start: Initializing Rix Core...")
        try:
            # Crucial: For Cloud Run, initialization.py should NOT rely on a local SA key file
            # if the service has a runtime service account. It should default to ADC.
            # We might need to modify initialization.py to have a 'cloud_run_mode'
            # or ensure config_manager.get_google_app_credentials() returns None in this env.
            # For now, assuming initialization.initialize_core() can run using ADC for Vertex AI.
            init_success = initialization.initialize_core(
                # Example of how we might control initialization for specific services:
                # skip_sql_pool=True,
                # skip_firestore_client=True, # If manager only needs Vertex AI
                # use_adc_implicitly=True # A hypothetical flag for initialization.py
            )
            if not init_success:
                error_msg = RixState.initialization_error or "Unknown Rix Core initialization error."
                logger.critical(f"Rix Core initialization FAILED in Manager Service: {error_msg}")
                SERVICE_STARTUP_ERROR = f"Rix Core Init Failed: {error_msg}"
                return # Stop further startup processing

            logger.info("Rix Core initialized successfully in Manager Service.")
            RIX_CORE_INITIALIZED = True

            # Load the manager soul prompt
            soul_file_name = rix_config.get_config("MANAGER_SOUL_FILE", "manager_v3_21.json")
            
            # Path to souls directory, assuming Rix_Brain is at the root of the deployment package
            souls_dir_path = RIX_BRAIN_PATH / "agents" / "souls"
            soul_path = souls_dir_path / soul_file_name

            if soul_path.is_file():
                MANAGER_SOUL_PROMPT_TEMPLATE = rix_utils.load_soul_prompt(soul_path)
                if MANAGER_SOUL_PROMPT_TEMPLATE:
                    logger.info(f"Manager soul prompt '{soul_file_name}' loaded successfully. Length: {len(MANAGER_SOUL_PROMPT_TEMPLATE)}")
                else:
                    SERVICE_STARTUP_ERROR = f"Failed to load content from manager soul prompt: {soul_path}"
                    logger.error(SERVICE_STARTUP_ERROR)
            else:
                SERVICE_STARTUP_ERROR = f"Manager soul file not found at: {soul_path}"
                logger.error(SERVICE_STARTUP_ERROR)
        
        except Exception as e:
            SERVICE_STARTUP_ERROR = f"Critical startup exception: {type(e).__name__} - {e}"
            logger.error(SERVICE_STARTUP_ERROR, exc_info=True)

# --- 6. API Endpoint ---
@app.post("/process_manager_task", response_model=ManagerTaskResponse)
async def process_manager_task(request: ManagerTaskRequest, raw_request: FastAPIRequest):
    logger.info(f"Manager Service /process_manager_task called. Session: {request.session_id}, Stage: {request.current_processing_stage}")
    logger.debug(f"Full request payload: {request.model_dump_json(indent=2)}")

    if SERVICE_STARTUP_ERROR:
        logger.error(f"Service startup error prevents processing: {SERVICE_STARTUP_ERROR}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {SERVICE_STARTUP_ERROR}")
    if not RIX_CORE_INITIALIZED or not vertex_llm or not MANAGER_SOUL_PROMPT_TEMPLATE:
        logger.error("Manager Service not properly initialized (core, LLM service, or soul missing).")
        raise HTTPException(status_code=503, detail="Service not properly initialized.")

    # Prepare context fields for filling placeholders in the soul prompt
    # Based on "input_context_fields_expected" in manager_v3_21.json
    context_fields_for_llm: Dict[str, Any] = {
        "session_id": request.session_id,
        "current_processing_stage": request.current_processing_stage,
        "user_input": request.user_input or "", # Ensure not None
        "history_context": request.history_context or [], # Ensure not None
        "thinker_report_content": request.thinker_report_content or "", # Ensure not None
        "classifier_output": request.classifier_output or {}, # Ensure not None
        "original_user_input": request.original_user_input or "" # Ensure not None
    }
    
    # Define invocation role for logging/debugging in vertex_llm
    invocation_role = f"ManagerService-{request.current_processing_stage}"

    try:
        # Call the LLM using the generic function from vertex_llm.py
        # This function needs to be added to your vertex_llm.py
        raw_llm_json_response = await vertex_llm.invoke_llm_for_json_action( # Making it async if vertex_llm supports async
            session_id=request.session_id,
            model_name_key="MANAGER_MODEL", # From config.json
            temperature_key="MANAGER_TEMPERATURE", # From config.json
            soul_prompt_template=MANAGER_SOUL_PROMPT_TEMPLATE,
            invocation_role=invocation_role,
            context_fields=context_fields_for_llm,
            history_context=request.history_context # Pass history separately for conversion
        )

        if not raw_llm_json_response:
            logger.error(f"LLM returned empty or None response for stage: {request.current_processing_stage}")
            # Fallback: Ask user for clarification
            return ManagerTaskResponse(
                action="ask_user_for_clarification",
                question_to_user="I seem to be having a bit of trouble processing that. Could you try rephrasing or providing more details?"
            )

        # Parse the LLM's JSON string response
        try:
            action_json = json.loads(raw_llm_json_response)
            if not isinstance(action_json, dict) or "action" not in action_json:
                logger.error(f"LLM response is not a valid JSON object with an 'action' field. Response: {raw_llm_json_response[:500]}")
                raise ValueError("Invalid action JSON structure from LLM")
            
            logger.info(f"Manager LLM action received: {action_json.get('action')}")
            logger.debug(f"Full action JSON from LLM: {action_json}")
            # The ManagerTaskResponse will automatically pick up all fields from action_json
            # due to Config.extra = "allow"
            return ManagerTaskResponse(**action_json)

        except json.JSONDecodeError as e_json:
            logger.error(f"Failed to parse LLM JSON response: {e_json}. Raw response: {raw_llm_json_response[:500]}...")
            # Fallback: Ask user for clarification
            return ManagerTaskResponse(
                action="ask_user_for_clarification",
                question_to_user="I received an unusual response from my internal processing. Could you please try your request again?"
            )
        except ValueError as e_val: # For our custom "Invalid action JSON structure"
             logger.error(str(e_val))
             return ManagerTaskResponse(
                action="ask_user_for_clarification",
                question_to_user="My internal decision format was incorrect. Could you try your request again, perhaps simplifying it?"
            )


    except LLMInvocationError as e_llm:
        logger.error(f"LLMInvocationError in Manager Service: {e_llm}", exc_info=True)
        # Fallback: Ask user for clarification, including error hint if appropriate
        return ManagerTaskResponse(
            action="ask_user_for_clarification",
            question_to_user=f"I encountered an issue with my thinking process ({e_llm.role} error). Could you please rephrase your request?"
        )
    except Exception as e:
        logger.error(f"Unexpected error in /process_manager_task: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {type(e).__name__}")


@app.get("/")
async def root_status_check(request: FastAPIRequest): # Added request to potentially log client IP
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Root path '/' accessed by {client_host} - Manager Service (V60.1) is alive.")
    if SERVICE_STARTUP_ERROR:
        return {"message": f"Rix Manager Service (V60.1) is running WITH STARTUP ERRORS: {SERVICE_STARTUP_ERROR}"}
    elif not RIX_CORE_INITIALIZED or not MANAGER_SOUL_PROMPT_TEMPLATE:
        return {"message": "Rix Manager Service (V60.1) is running BUT NOT FULLY INITIALIZED."}
    return {"message": "Rix Manager Service (V60.1 - Intelligent User Interface & Dispatcher) is running and healthy."}

# For local testing (optional):
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Rix Manager Service locally on port 8080...")
#     # This local run would require Rix_Brain and config.json to be in the correct
#     # relative paths from this script, or sys.path adjusted accordingly.
#     uvicorn.run(app, host="0.0.0.0", port=8080)

# --- END OF FILE rix-manager-service/main.py ---
