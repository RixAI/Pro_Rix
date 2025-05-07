# --- START OF FILE Pro_Rix/test_manager_logic.py ---
import sys
import os
import json
import asyncio # For running async functions if vertex_llm uses them
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional

# --- Configure Logging ---
# (Using a simple config for the test script)
log_format = '%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, force=True)
log = logging.getLogger("TestManagerScript")
# Set Rix_Heart logging level (or other loggers) if needed
logging.getLogger("Rix_Heart").setLevel(logging.INFO) 
logging.getLogger("Rix_Brain").setLevel(logging.INFO) 

# --- Add Project Root to Path (similar to run_rix_cli.py) ---
try:
    # Assume this script is in Pro_Rix/
    PROJECT_ROOT = Path(__file__).resolve().parent 
    RIX_BRAIN_PATH = PROJECT_ROOT / "Rix_Brain"
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    log.info(f"Added {PROJECT_ROOT} to sys.path")
except Exception as path_e:
    log.error(f"FATAL PATH ERROR: {path_e}", exc_info=True)
    sys.exit(1)

# --- Import Rix Brain Modules ---
log.info("Importing Rix Brain modules...")
try:
    from Rix_Brain.core import initialization
    from Rix_Brain.core import config_manager as rix_config
    from Rix_Brain.core import utils as rix_utils
    from Rix_Brain.services import vertex_llm # Needs invoke_llm_for_json_action
    from Rix_Brain.core import global_state as RixState
    from Rix_Brain.core.exceptions import LLMInvocationError
    log.info("Rix Brain modules imported successfully.")
except ImportError as e:
    log.error(f"FATAL IMPORT ERROR: {e}. Make sure Rix_Brain is in the Python path and all dependencies are installed in .venv.", exc_info=True)
    sys.exit(1)
except Exception as e:
    log.error(f"Unexpected import error: {e}", exc_info=True)
    sys.exit(1)

# --- Main Test Function ---
async def run_test():
    log.info("--- Starting Local Manager Logic Test ---")

    # 1. Ensure Rix Core is Initialized (Locally)
    log.info("Initializing Rix Core locally...")
    # Use local ADC or SA Key file based on config_manager's logic
    # Ensure config_manager V1.5.0 (using ENV VAR fallback) is used
    # If RIX_PROJECT_ROOT env var isn't set locally, config_manager should use Path(__file__) fallback.
    if not hasattr(initialization, 'initialize_core'):
        log.error("Imported 'initialization' module seems invalid (missing initialize_core).")
        return
        
    init_success = initialization.initialize_core()
    if not init_success:
        init_error = getattr(RixState, 'initialization_error', 'Unknown Init Error')
        log.error(f"Rix Core initialization failed: {init_error}")
        return
    log.info("Rix Core initialized successfully.")
    
    # Check if vertex_llm function exists
    if not hasattr(vertex_llm, 'invoke_llm_for_json_action'):
         log.error("Function 'invoke_llm_for_json_action' not found in Rix_Brain.services.vertex_llm.py. Please add it.")
         return

    # 2. Load Manager Soul Prompt
    log.info("Loading Manager soul prompt...")
    try:
        soul_file_name = rix_config.get_config("MANAGER_SOUL_FILENAME", "manager_v3_21.json")
        # Use config_manager's path logic (which should use RIX_PROJECT_ROOT env var or fallback)
        # We need a robust way to get the souls dir path from config_manager
        # Let's assume config_manager.RIX_BRAIN_DIR is correctly determined:
        souls_dir_path = Path(rix_config.RIX_BRAIN_DIR) / "agents" / "souls"
        soul_path = souls_dir_path / soul_file_name
        
        if not soul_path.is_file():
             log.error(f"Manager soul file not found at derived path: {soul_path}")
             return

        manager_soul_template = rix_utils.load_soul_prompt(soul_path)
        if not manager_soul_template:
            log.error(f"Failed to load content from manager soul prompt: {soul_path}")
            return
        log.info(f"Manager soul prompt '{soul_file_name}' loaded. Length: {len(manager_soul_template)}")
    except Exception as e:
        log.error(f"Error loading soul prompt: {e}", exc_info=True)
        return

    # 3. Prepare Mock Input Data
    log.info("Preparing mock input data...")
    session_id = f"local_test_{rix_utils.get_current_timestamp()}"
    # --- Test Case 1: Cartoon Request ---
    user_input_cartoon = "Rix, please make a cartoon of Ravi finding a glowing fruit in the jungle."
    history_context_cartoon = [
        # Add a few dummy history entries if needed by the soul prompt or history converter
        {"actor": "User", "message_type": "user_input", "content": "Hi Rix"},
        {"actor": "Manager", "message_type": "system_response", "content": "Hello Vishal!"}
    ]
    context_fields_cartoon = {
        "session_id": session_id,
        "current_processing_stage": "INITIAL_USER_INPUT",
        "user_input": user_input_cartoon,
        "history_context": history_context_cartoon, # Note: history is also passed separately usually
        "thinker_report_content": None, 
        "classifier_output": None, 
        "original_user_input": user_input_cartoon 
    }
    
    # --- Test Case 2: Simple Chat ---
    user_input_chat = "Just saying hello."
    history_context_chat = history_context_cartoon # Reuse simple history
    context_fields_chat = {
        "session_id": session_id + "_chat",
        "current_processing_stage": "INITIAL_USER_INPUT",
        "user_input": user_input_chat,
        "history_context": history_context_chat,
        "thinker_report_content": None, 
        "classifier_output": None, 
        "original_user_input": user_input_chat 
    }
    
    # --- Test Case 3: Finalize Report ---
    thinker_report_content = {
        "summary": "Cartoon scene 'Ravi fruit' (scene_abc123) completed fake generation. Final video at /fake/final.mp4",
        "status": "success"
    }
    history_context_report = history_context_cartoon + [
        {"actor": "Manager", "message_type": "dispatch_instruction", "content": "@Thinker handle cartoon ravi fruit"},
        {"actor": "Thinker", "message_type": "internal_report", "content": json.dumps(thinker_report_content)}
    ]
    context_fields_report = {
        "session_id": session_id + "_report",
        "current_processing_stage": "FINALIZE_THINKER_REPORT",
        "user_input": None, # No new user input for this stage
        "history_context": history_context_report, 
        "thinker_report_content": thinker_report_content, 
        "classifier_output": None, 
        "original_user_input": user_input_cartoon # Need original request context potentially
    }

    # 4. Run Test Cases
    test_cases = [
        ("Cartoon Request", context_fields_cartoon, history_context_cartoon),
        ("Simple Chat", context_fields_chat, history_context_chat),
        ("Finalize Report", context_fields_report, history_context_report)
    ]

    for name, context_fields, history in test_cases:
        log.info(f"\n--- Running Test Case: {name} ---")
        try:
            log.info(f"Context fields being sent to LLM function: {json.dumps(context_fields, indent=2)}")
            log.info(f"History context being sent: {json.dumps(history, indent=2)}")
            
            raw_json_output = await vertex_llm.invoke_llm_for_json_action(
                session_id=context_fields["session_id"],
                model_name_key="MANAGER_MODEL", 
                temperature_key="MANAGER_TEMPERATURE",
                soul_prompt_template=manager_soul_template,
                invocation_role=f"TestManagerScript-{context_fields['current_processing_stage']}",
                context_fields=context_fields,
                history_context=history
            )

            if raw_json_output:
                log.info(f"Raw LLM Response for '{name}':\n{raw_json_output}")
                try:
                    # Try parsing the JSON to validate
                    parsed_action = json.loads(raw_json_output)
                    log.info(f"Parsed Action JSON for '{name}':\n{json.dumps(parsed_action, indent=2)}")
                    if "action" not in parsed_action:
                         log.warning(f"VALIDATION WARNING for '{name}': 'action' key missing in response JSON.")
                except Exception as e_parse:
                    log.error(f"JSON Parsing Error for '{name}' response: {e_parse}")
            else:
                log.warning(f"LLM returned None or empty response for '{name}'.")

        except LLMInvocationError as e_llm:
            log.error(f"LLMInvocationError during '{name}' test: {e_llm}", exc_info=True)
        except Exception as e:
            log.error(f"Unexpected error during '{name}' test: {e}", exc_info=True)
        log.info(f"--- Finished Test Case: {name} ---")

    log.info("--- Local Manager Logic Test Finished ---")

# --- Run the async test function ---
if __name__ == "__main__":
    # Ensure dependencies are installed in the active .venv
    log.info(f"Running test script. Ensure .venv is active and dependencies from")
    log.info(f"cloud_services/rix-manager-service/requirements.txt are installed.")
    # Example: pip install -r cloud_services/rix-manager-service/requirements.txt
    
    asyncio.run(run_test())

# --- END OF FILE Pro_Rix/test_manager_logic.py ---