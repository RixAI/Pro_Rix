# --- START OF FILE cloud_services/rix-manager-service/main.py ---
# Version: V60.1 (Intelligent Entry Point & Dispatcher - Env Var Pathing)

import sys
import os
from pathlib import Path
import logging
import json # Make sure json is imported
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Body, Request as FastAPIRequest
from pydantic import BaseModel, Field

# --- 1. Configure Logging Early ---
# BasicConfig should be called only once. If Rix_Brain modules also call it,
# it might lead to unexpected behavior. Ideally, configure logging at the
# highest entry point or ensure subsequent calls don't override.
# For a service, this main.py is the highest entry point.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),  # Allow log level from env
    format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s',
    force=True # Ensure this config takes precedence if other modules try to basicConfig
)
logger = logging.getLogger(__name__)

# Log the RIX_PROJECT_ROOT environment variable for confirmation during startup
RIX_PROJECT_ROOT_ENV = os.getenv('RIX_PROJECT_ROOT')
logger.info(f"RIX_PROJECT_ROOT environment variable at service start = {RIX_PROJECT_ROOT_ENV}")
if not RIX_PROJECT_ROOT_ENV:
    logger.critical("CRITICAL: RIX_PROJECT_ROOT environment variable is NOT SET. Rix_Brain imports will likely fail.")
    # Service will likely fail to initialize properly.

# --- 2. Rix Brain Core Imports ---
# Relies on RIX_PROJECT_ROOT being /app (set in Dockerfile ENV)
# and Rix_Brain being copied to /app/Rix_Brain by Dockerfile.
# config_manager.py will use RIX_PROJECT_ROOT to find config.json at /app/config.json
SERVICE_STARTUP_ERROR: Optional[str] = None # Global to store critical startup errors
initialization = None # Initialize to None
rix_config = None
rix_utils = None
vertex_llm = None
RixState = None
LLMInvocationError = None

try:
    logger.info("Attempting to import Rix_Brain modules...")
    from Rix_Brain.core import initialization
    from Rix_Brain.core import config_manager as rix_config
    from Rix_Brain.core import utils as rix_utils
    from Rix_Brain.services import vertex_llm # Ensure invoke_llm_for_json_action is added here
    from Rix_Brain.core import global_state as RixState
    from Rix_Brain.core.exceptions import LLMInvocationError
    
    # Check if imports actually loaded modules
    if not all([initialization, rix_config, rix_utils, vertex_llm, RixState, LLMInvocationError]):
        raise ImportError("One or more Rix_Brain modules failed to import correctly (resolved to None).")
    logger.info("Successfully imported Rix_Brain modules for Manager Service.")

except ImportError as e:
    logger.error(f"FATAL: Failed to import Rix_Brain modules in Manager Service: {e}", exc_info=True)
    SERVICE_STARTUP_ERROR = f"ImportError for Rix_Brain: {e}. Check Dockerfile COPY and RIX_PROJECT_ROOT ENV."
    # Set modules to None so startup_event can check them
    initialization = None; rix_config = None; rix_utils = None; vertex_llm = None; RixState = None; LLMInvocationError = type('LLMInvocationError', (Exception,), {})


# --- 3. FastAPI App Setup ---
app = FastAPI(
    title="Rix Manager Service",
    description="V60.1 - Intelligent User Interface ('Rix' Persona) and Task Dispatcher.",
    version="60.1.0"
)

# --- 4. Global State for Initialization & Soul Prompt ---
RIX_CORE_INITIALIZED = False
MANAGER_SOUL_PROMPT_TEMPLATE: Optional[str] = None
# SERVICE_STARTUP_ERROR is defined above

# --- 5. Pydantic Models for API --- (Same as before)
class ManagerTaskRequest(BaseModel):
    session_id: str
    current_processing_stage: str = Field(description="e.g., 'INITIAL_USER_INPUT', 'FINALIZE_THINKER_REPORT'")
    user_input: Optional[str] = None
    history_context: List[Dict[str, Any]] = Field(default_factory=list)
    thinker_report_content: Optional[Any] = None 
    classifier_output: Optional[Dict[str, Any]] = None
    original_user_input: Optional[str] = None

class ManagerTaskResponse(BaseModel):
    action: str
    class Config: extra = "allow"

# --- 6. Startup Event for Initialization ---
@app.on_event("startup")
async def startup_event():
    global RIX_CORE_INITIALIZED, MANAGER_SOUL_PROMPT_TEMPLATE, SERVICE_STARTUP_ERROR

    if SERVICE_STARTUP_ERROR: # If imports already failed
        logger.error(f"Skipping Rix Core initialization due to prior import errors: {SERVICE_STARTUP_ERROR}")
        return

    if not RIX_CORE_INITIALIZED:
        logger.info("Manager Service Startup: Initializing Rix Core...")
        try:
            if not initialization or not RixState or not rix_config or not rix_utils:
                 SERVICE_STARTUP_ERROR = "Core Rix_Brain modules not imported, cannot initialize."
                 logger.critical(SERVICE_STARTUP_ERROR)
                 return

            # For Cloud Run, initialization.py should use ADC.
            # Ensure config_manager.get_google_app_credentials() returns None in Cloud Run.
            init_success = initialization.initialize_core() # Add flags like skip_sql=True if needed
            
            if not init_success:
                error_msg = getattr(RixState, 'initialization_error', "Unknown Rix Core initialization error.")
                logger.critical(f"Rix Core initialization FAILED in Manager Service: {error_msg}")
                SERVICE_STARTUP_ERROR = f"Rix Core Init Failed: {error_msg}"
                return

            logger.info("Rix Core initialized successfully in Manager Service.")
            RIX_CORE_INITIALIZED = True

            soul_file_name = rix_config.get_config("MANAGER_SOUL_FILENAME", "manager_v3_21.json")
            # config_manager now uses RIX_PROJECT_ROOT to build its internal paths
            # So, get_souls_dir_path() should work if defined in config_manager
            # For simplicity here, let's assume config_manager.RIX_BRAIN_DIR is correct
            # and we construct the path directly.
            # More robust: souls_dir_path = rix_config.get_souls_dir_path()
            souls_dir_path = Path(rix_config.RIX_BRAIN_DIR) / "agents" / "souls"
            soul_path = souls_dir_path / soul_file_name

            if soul_path.is_file():
                MANAGER_SOUL_PROMPT_TEMPLATE = rix_utils.load_soul_prompt(soul_path)
                if MANAGER_SOUL_PROMPT_TEMPLATE:
                    logger.info(f"Manager soul prompt '{soul_file_name}' loaded. Length: {len(MANAGER_SOUL_PROMPT_TEMPLATE)}")
                else:
                    SERVICE_STARTUP_ERROR = f"Failed to load content from manager soul: {soul_path}"
                    logger.error(SERVICE_STARTUP_ERROR)
            else:
                SERVICE_STARTUP_ERROR = f"Manager soul file not found: {soul_path}"
                logger.error(SERVICE_STARTUP_ERROR)
        
        except Exception as e:
            SERVICE_STARTUP_ERROR = f"Critical exception during startup_event: {type(e).__name__} - {e}"
            logger.error(SERVICE_STARTUP_ERROR, exc_info=True)

# --- 7. API Endpoint ---
@app.post("/process_manager_task", response_model=ManagerTaskResponse)
async def process_manager_task(request: ManagerTaskRequest): # Removed raw_request for now
    logger.info(f"Manager Service /process_manager_task. Session: {request.session_id}, Stage: {request.current_processing_stage}")
    logger.debug(f"Request payload: {request.model_dump_json(indent=2)}")

    if SERVICE_STARTUP_ERROR:
        logger.error(f"Cannot process: Service has startup errors: {SERVICE_STARTUP_ERROR}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {SERVICE_STARTUP_ERROR}")
    if not RIX_CORE_INITIALIZED or not vertex_llm or not MANAGER_SOUL_PROMPT_TEMPLATE or not LLMInvocationError:
        logger.error("Cannot process: Manager Service not properly initialized (core, LLM, soul, or exception type missing).")
        raise HTTPException(status_code=503, detail="Service not properly initialized (missing components).")

    context_fields_for_llm: Dict[str, Any] = {
        "session_id": request.session_id,
        "current_processing_stage": request.current_processing_stage,
        "user_input": request.user_input or "", 
        "history_context": request.history_context or [], 
        "thinker_report_content": request.thinker_report_content or "", 
        "classifier_output": request.classifier_output or {}, 
        "original_user_input": request.original_user_input or "" 
    }
    invocation_role = f"ManagerService-{request.current_processing_stage}"

    try:
        if not hasattr(vertex_llm, 'invoke_llm_for_json_action'):
            logger.error("vertex_llm module does not have 'invoke_llm_for_json_action' function.")
            raise HTTPException(status_code=500, detail="LLM action function missing in service.")

        raw_llm_json_response = await vertex_llm.invoke_llm_for_json_action(
            session_id=request.session_id, model_name_key="MANAGER_MODEL",
            temperature_key="MANAGER_TEMPERATURE", soul_prompt_template=MANAGER_SOUL_PROMPT_TEMPLATE,
            invocation_role=invocation_role, context_fields=context_fields_for_llm,
            history_context=request.history_context
        )

        if not raw_llm_json_response:
            logger.error(f"LLM returned empty for stage: {request.current_processing_stage}. Fallback action.")
            return ManagerTaskResponse(action="ask_user_for_clarification", question_to_user="I had trouble processing that. Could you rephrase?")

        try:
            action_json = json.loads(raw_llm_json_response)
            if not isinstance(action_json, dict) or "action" not in action_json:
                logger.error(f"LLM response not valid JSON with 'action'. Raw: {raw_llm_json_response[:500]}")
                raise ValueError("Invalid action JSON from LLM")
            logger.info(f"Manager LLM action: {action_json.get('action')}")
            return ManagerTaskResponse(**action_json)
        except (json.JSONDecodeError, ValueError) as e_parse:
            logger.error(f"Failed to parse LLM JSON: {e_parse}. Raw: {raw_llm_json_response[:500]}...", exc_info=True)
            return ManagerTaskResponse(action="ask_user_for_clarification", question_to_user="My internal response format was unusual. Please try again.")

    except LLMInvocationError as e_llm:
        logger.error(f"LLMInvocationError in Manager Service: {e_llm}", exc_info=True)
        return ManagerTaskResponse(action="ask_user_for_clarification", question_to_user=f"Issue with my thinking ({e_llm.role} error). Please rephrase.")
    except Exception as e:
        logger.error(f"Unexpected error in /process_manager_task: {type(e).__name__} - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected server error processing manager task: {type(e).__name__}")

@app.get("/")
async def root_status_check(request: FastAPIRequest): 
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Root path '/' accessed by {client_host} - Manager Service (V60.1) is alive check.")
    if SERVICE_STARTUP_ERROR: return {"message": f"Rix Manager Service (V60.1) running WITH STARTUP ERRORS: {SERVICE_STARTUP_ERROR}"}
    if not RIX_CORE_INITIALIZED or not MANAGER_SOUL_PROMPT_TEMPLATE: return {"message": "Rix Manager Service (V60.1) running BUT NOT FULLY INITIALIZED."}
    return {"message": "Rix Manager Service (V60.1 - Intelligent User Interface & Dispatcher) is running and healthy."}

# For local testing:
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Rix Manager Service locally on port specified by Uvicorn (e.g. 8000 or 8080)...")
#     uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True) # Use PORT from env for Cloud Run
# --- END OF FILE cloud_services/rix-manager-service/main.py ---