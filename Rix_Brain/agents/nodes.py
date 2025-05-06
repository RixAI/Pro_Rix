# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\agents\nodes.py ---
# Version: 1.1.0 (V54.0 - ClassifierNode HTTP Implemented, Finalizer Named Correctly)
# Author: Vishal Sharma / Gemini
# Date: May 7, 2025
# Description: Implements HTTP call for classifier_node. Other nodes are placeholders.

import logging
import json
import datetime # For thinker_node placeholder tool_req_id
from typing import Dict, Any, Literal, Optional, List, Tuple

# --- Library for HTTP calls ---
import httpx # For making async HTTP requests
import asyncio # For running async code if needed (httpx.AsyncClient is async)

# --- Rix Core Imports ---
# Import RixState for config and utils for OIDC token
from Rix_Brain.core import global_state as RixState
from Rix_Brain.core import utils as rix_utils # For get_oidc_token
# Import config manager directly if RixState.config isn't reliably populated at node execution
from Rix_Brain.core import config_manager as rix_config


# --- Logging ---
logger = logging.getLogger("Rix_Brain.agents")

# --- Placeholder for PipelineState (defined in graph_pipeline.py) ---
if 'PipelineState' not in locals():
    from typing import TypedDict
    class PipelineState(TypedDict, total=False):
        session_id: str; user_input: Optional[str]; classification: Optional[Literal["CHAT", "ASK", "WORK", "RCD"]]; dispatch_instruction: Optional[str]; current_tool_request: Optional[Dict]; current_tool_result: Optional[Dict]; agent_history: List[Tuple[str, str]]; intermediate_steps: List[Any]; final_response: Optional[str]; error_message: Optional[str]; rcd_plan: Optional[List[Dict]]; rcd_current_step_index: int; rcd_step_outputs: Dict[int, Any]; next_node_target: Optional[str]


# --- NODE TIMEOUT CONFIGURATION ---
DEFAULT_NODE_HTTP_TIMEOUT = 30.0  # seconds

async def classifier_node(state: PipelineState) -> Dict[str, Any]: # Made async
    """
    Classifier Agent Node.
    Makes an HTTP call to the rix-classifier-service on Cloud Run.
    Falls back to placeholder logic if the HTTP call fails.
    """
    session_id = state.get("session_id", "unknown_session")
    user_input = state.get("user_input", "")
    log_prefix = f"[{session_id}] ClassifierNode"
    logger.info(f"{log_prefix}: Entered. Input: '{user_input[:100]}...'")

    service_url = rix_config.get_config("RIX_CLASSIFIER_SERVICE_URL")

    if not service_url:
        logger.error(f"{log_prefix}: RIX_CLASSIFIER_SERVICE_URL not found in config. Falling back to placeholder.")
        classification_result = "CHAT"
        if "help" in user_input.lower() or "what can you do" in user_input.lower(): classification_result = "ASK"
        elif "file" in user_input.lower() or "create" in user_input.lower() or "run" in user_input.lower(): classification_result = "WORK"
        logger.info(f"{log_prefix}: Placeholder classification: {classification_result}")
        return {"classification": classification_result, "error_message": "Service URL missing, used placeholder."}

    payload = {"session_id": session_id, "user_input": user_input}
    classification_from_service = None
    error_from_service = None
    http_call_succeeded = False

    try:
        logger.info(f"{log_prefix}: Attempting HTTP POST to {service_url}")
        token = rix_utils.get_oidc_token(target_audience_url=service_url)
        if not token:
            error_from_service = "Failed to generate OIDC token for classifier service."
            logger.error(f"{log_prefix}: {error_from_service}")
        else:
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=DEFAULT_NODE_HTTP_TIMEOUT) as client:
                response = await client.post(service_url, json=payload, headers=headers)
            logger.info(f"{log_prefix}: Received response from service. Status: {response.status_code}")
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"{log_prefix}: Service response data: {response_data}")
                classification_from_service = response_data.get("classification")
                if not classification_from_service:
                    error_from_service = "Classifier service responded 200 but classification missing."
                    logger.error(f"{log_prefix}: {error_from_service} - Data: {response_data}")
                else: http_call_succeeded = True
            else:
                error_from_service = f"Classifier service error {response.status_code}. Response: {response.text[:200]}"
                logger.error(f"{log_prefix}: {error_from_service}")
    except httpx.TimeoutException: error_from_service = f"Request to classifier service timed out."; logger.error(f"{log_prefix}: {error_from_service}")
    except httpx.RequestError as e: error_from_service = f"HTTP request error to classifier: {e}"; logger.error(f"{log_prefix}: {error_from_service}", exc_info=True)
    except json.JSONDecodeError as e: error_from_service = f"Failed to decode JSON from classifier: {e}"; logger.error(f"{log_prefix}: {error_from_service}. Raw: {response.text[:200] if 'response' in locals() else 'N/A'}")
    except Exception as e: error_from_service = f"Unexpected error in classifier call: {e}"; logger.error(f"{log_prefix}: {error_from_service}", exc_info=True)

    final_classification = None; current_error_message = state.get("error_message")
    if http_call_succeeded and classification_from_service:
        final_classification = classification_from_service; current_error_message = None
        logger.info(f"{log_prefix}: Using classification from service: {final_classification}")
    else:
        logger.warning(f"{log_prefix}: HTTP call failed or URL missing. Fallback. Error: {error_from_service}")
        final_classification = "CHAT"
        if "help" in user_input.lower(): final_classification = "ASK"
        elif "file" in user_input.lower(): final_classification = "WORK"
        logger.info(f"{log_prefix}: Placeholder classification: {final_classification}")
        if error_from_service: current_error_message = f"{current_error_message or ''} | Classifier Service Call Failed: {error_from_service}".strip(" | ")
    update_to_state = {"classification": final_classification, "error_message": current_error_message}
    logger.info(f"{log_prefix}: Exiting. Updating state: {update_to_state}")
    return update_to_state

# --- Other placeholder nodes ---
# (These are simplified versions from response #40, assuming the detailed versions from #46 for other nodes are too verbose for this step)
import uuid # For placeholder memory_writer_node

async def thinker_node(state: PipelineState) -> Dict[str, Any]: # Made async for consistency
    session_id = state.get("session_id", "unknown_session"); classification = state.get("classification", "UNKNOWN")
    user_input = state.get("user_input", ""); log_prefix = f"[{session_id}] ThinkerNode"
    logger.info(f"{log_prefix}: Entered. Classification: {classification}, Input: '{user_input[:50]}...'")
    current_tool_request = None; final_response_from_thinker = None
    if classification == "WORK":
        logger.info(f"{log_prefix}: Placeholder: Simulating list_files tool request for WORK.")
        current_tool_request = {"name": "list_files", "args": {"path": "Rix_Brain/agents/souls"}}
    elif classification == "ASK":
        logger.info(f"{log_prefix}: Placeholder: Simulating direct answer for ASK.")
        final_response_from_thinker = f"Placeholder: Thinking about your ASK on '{user_input[:30]}...'"
    update_to_state = {"current_tool_request": current_tool_request, "error_message": None}
    if final_response_from_thinker: update_to_state["final_response"] = final_response_from_thinker
    logger.info(f"{log_prefix}: Exiting. Updates: {update_to_state}")
    return update_to_state

async def tool_executor_node(state: PipelineState) -> Dict[str, Any]: # Made async
    session_id = state.get("session_id", "unknown_session"); tool_request = state.get("current_tool_request")
    log_prefix = f"[{session_id}] ToolExecutorNode"; logger.info(f"{log_prefix}: Entered.")
    current_tool_result = None; error_message = None
    if tool_request and isinstance(tool_request, dict):
        tool_name = tool_request.get("name"); tool_args = tool_request.get("args", {})
        logger.info(f"{log_prefix}: Received tool request for: {tool_name}")
        logger.warning(f"{log_prefix}: V54.0 - Using direct local tool call placeholder.")
        try:
            from Rix_Brain.tools import tool_manager # Direct import for local call
            execution_result = tool_manager.execute_tool(str(tool_name), dict(tool_args), session_id)
            current_tool_result = {"command": tool_name, "status": execution_result.get("status", "unknown"), "result": execution_result.get("result"), "message": execution_result.get("message", "")}
            logger.info(f"{log_prefix}: Local tool '{tool_name}' placeholder exec status: {current_tool_result.get('status')}")
            if execution_result.get("status") == "failed": error_message = execution_result.get("message") or f"Tool '{tool_name}' failed."
        except Exception as e: error_message = f"Local tool exec placeholder error: {e}"; logger.exception(error_message); current_tool_result = {"command": tool_name, "status": "failed", "error": {"message": error_message}}
    else: error_message = "No valid tool_request in state."; logger.warning(f"{log_prefix}: {error_message}")
    update_to_state = {"current_tool_request": None, "current_tool_result": current_tool_result, "error_message": error_message or state.get("error_message")}
    logger.info(f"{log_prefix}: Exiting. Result status: {current_tool_result.get('status') if current_tool_result else 'N/A'}")
    return update_to_state

async def memory_writer_node(state: PipelineState) -> Dict[str, Any]: # Made async
    session_id = state.get("session_id", "unknown_session"); log_prefix = f"[{session_id}] MemoryWriterNode"; logger.info(f"{log_prefix}: Entered.")
    logger.warning(f"{log_prefix}: Placeholder - Simulating memory write.")
    update_to_state = {"last_memory_id": str(uuid.uuid4()), "error_message": None }
    logger.info(f"{log_prefix}: Exiting. Updates: {update_to_state}")
    return update_to_state

async def finalizer_node(state: PipelineState) -> Dict[str, Any]: # Made async, name standardized
    session_id = state.get("session_id", "unknown_session"); log_prefix = f"[{session_id}] FinalizerNode"; logger.info(f"{log_prefix}: Entered.")
    current_final_response = state.get("final_response"); error_message = state.get("error_message")
    user_input = state.get("user_input"); classification = state.get("classification"); tool_result = state.get("current_tool_result")
    logger.warning(f"{log_prefix}: Placeholder logic.")
    final_response_to_user = ""
    if error_message: final_response_to_user = f"Issue: {error_message}."
    elif current_final_response: final_response_to_user = str(current_final_response)
    elif classification == "CHAT": final_response_to_user = f"Chat about '{str(user_input)[:20]}...' processed (Placeholder)"
    elif tool_result and tool_result.get("status") == "success": payload = tool_result.get("result", "Tool executed."); final_response_to_user = f"Tool '{tool_result.get('command')}' run. Result: {str(payload)[:50]}"
    else: final_response_to_user = f"Processed '{str(user_input)[:20]}...' (Generic placeholder)"
    update_to_state = {"final_response": final_response_to_user, "error_message": None}
    logger.info(f"{log_prefix}: Exiting. Final Response: {final_response_to_user[:70]}...")
    return update_to_state

print(f"--- Module Defined: {__name__} (V1.1.0 - Classifier HTTP, Finalizer Named Correctly) ---", flush=True)
# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\agents\nodes.py ---