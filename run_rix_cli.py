# --- START OF FILE C:\Rix_Dev\Pro_Rix\run_rix_cli.py ---
# Version: 1.2.3 (V53.0 - Async LangGraph Invocation)

import sys
import os
from pathlib import Path
import datetime
import logging
import json
import time
from typing import Dict, Any, Optional
import asyncio # <--- IMPORT ASYNCIO

print("--- Rix CLI Chat Interface (V1.2.3 - Async Graph Invoke) ---", flush=True)

# --- 1. Setup Paths & Logging --- (Keep as is)
try:
    script_dir = Path(__file__).resolve().parent; project_root = script_dir; paths_to_add = [str(project_root)]
    print(f"Attempting to add to sys.path: {paths_to_add}", flush=True)
    for p in paths_to_add:
        if p not in sys.path: sys.path.insert(0, p); print(f"  Successfully Added: {p}", flush=True)
        else: print(f"  Already in path: {p}", flush=True)
except Exception as path_e: print(f"FATAL PATH ERROR: {path_e}", flush=True); sys.exit(1)
log_format = '%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format); log = logging.getLogger("RixCli")
logging.getLogger("Rix_Heart").setLevel(logging.WARNING); logging.getLogger("Rix_Brain").setLevel(logging.INFO)
logging.getLogger("langgraph").setLevel(logging.INFO)

# --- 2. Import Rix Brain Core & GRAPH GETTER --- (Keep as is)
try:
    log.info("Importing Rix Brain modules (V53.0)...")
    from Rix_Brain.core import initialization as rix_initialization
    from Rix_Brain.orchestration.graph_pipeline import get_compiled_graph, PipelineState
    log.info("Core Rix Brain modules and graph getter imported successfully.")
except ImportError as e: log.exception(f"FATAL IMPORT ERROR: {e}"); sys.exit(1)
except Exception as e: log.exception(f"FATAL UNEXPECTED IMPORT ERROR: {e}"); sys.exit(1)

# --- 3. Initialize Rix Brain --- (Keep as is)
log.info("Initializing Rix Brain Core for CLI...")
init_success = rix_initialization.initialize_core()
if not init_success:
    log.critical("HALTING CLI: Rix Brain initialization failed.")
    try: from Rix_Brain.core import global_state as RixState_fallback; init_error = getattr(RixState_fallback, 'initialization_error', 'Unknown Init Error'); log.critical(f"Init Error: {init_error}")
    except Exception: log.critical("Could not retrieve init error.")
    sys.exit(1)
log.info("Rix Brain Initialized Successfully for CLI.")

# --- 4. GET the Compiled Graph App --- (Keep as is)
log.info("Attempting to get compiled LangGraph app...")
rix_graph_app = get_compiled_graph()
if rix_graph_app is None:
    log.critical("HALTING CLI: LangGraph application `rix_graph_app` is None."); sys.exit(1)
log.info(f"LangGraph `rix_graph_app` obtained successfully (Type: {type(rix_graph_app)}).")


# --- 5. Main Chat Loop - NOW ASYNC ---
async def run_cli_async(): # <--- MADE ASYNC
    log.info("Starting Rix CLI main ASYNC loop (LangGraph V53.0)...")
    cli_session_id_base = f"cli_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nWelcome to Rix CLI (LangGraph V53.0 - Async)"); print(f"Base Session: {cli_session_id_base}"); print('Type message, or "quit"/"exit" to end.')
    turn_counter = 0
    while True:
        turn_counter += 1; current_turn_session_id = f"{cli_session_id_base}_turn{turn_counter}"; print("-" * 20)
        try:
            # User input is synchronous, so we can't easily await it inside this async loop directly
            # without more complex asyncio input handling. For CLI, it's often simpler
            # to keep the input part synchronous and then call the async graph part.
            # However, for a fully async main loop, one might use asyncio.to_thread for input.
            # For now, let's keep it simple. `input()` will block the event loop.
            user_input = input(f"[{current_turn_session_id}] You: ").strip()

            if not user_input: continue
            if user_input.lower() in ["quit", "exit"]: log.info("Exit command received."); break
            log.info(f"CLI Turn {turn_counter} (ID: {current_turn_session_id}): Processing: '{user_input[:100]}...'")
            start_time = time.monotonic()
            initial_graph_input: PipelineState = { "session_id": current_turn_session_id, "user_input": user_input, "agent_history": [], "intermediate_steps": [], "classification": None, "final_response": None, "error_message": None, "rcd_plan": None, "rcd_current_step_index": 0, "rcd_step_outputs": {}} # type: ignore
            graph_config = {"configurable": {"thread_id": current_turn_session_id}}
            final_state: Optional[PipelineState] = None; log.info(f"Invoking LangGraph ASYNC stream with config: {graph_config}"); print("Rix (streaming output):", flush=True)
            try:
                # --- USE ASYNC STREAM ---
                async for event_chunk in rix_graph_app.astream(initial_graph_input, config=graph_config): # <--- .astream()
                    for node_name, node_output_state in event_chunk.items():
                        log.info(f"--- Node '{node_name}' Completed ---")
                        final_state = node_output_state
                log.info("LangGraph async stream finished.")
                if final_state is None:
                    log.warning("Async Stream completed but final_state is None. Using ainvoke() as fallback.")
                    final_state = await rix_graph_app.ainvoke(initial_graph_input, config=graph_config) # <--- await .ainvoke()
            except Exception as graph_exec_e: log.exception(f"Error in LangGraph async call: {graph_exec_e}"); print(f"\nRix Error: Graph processing error: {graph_exec_e}"); continue
            finally: duration = time.monotonic() - start_time; log.info(f"Graph exec for turn {turn_counter} took {duration:.2f}s.")
            print("\n" + "="*5 + " Rix Response (Final State) " + "="*5)
            if final_state and isinstance(final_state, dict):
                response_to_display = final_state.get("final_response"); error_msg = final_state.get("error_message")
                if error_msg: print(f"Rix Error: {error_msg}"); log.error(f"Pipeline error: {error_msg}");
                elif response_to_display: print(f"Rix: {response_to_display}")
                else: print("Rix: [Graph completed, no specific response or error.]"); log.warning("Graph finished but no final_response.")
            else: print("Rix: [No final state dict received]"); log.error("No final state dict.")
        except EOFError: log.info("EOF received."); break
        except KeyboardInterrupt: log.info("Keyboard interrupt."); break
        except Exception as loop_e: log.exception(f"Unexpected CLI loop error: {loop_e}"); print(f"\nCLI Error: {loop_e}")
    print("\nRix CLI session ended.")

# --- 6. Run the CLI ---
if __name__ == "__main__":
    if not init_success: print("ERROR: Init failed.", file=sys.stderr); sys.exit(1)
    if rix_graph_app is None: print("ERROR: Graph app unavailable.", file=sys.stderr); sys.exit(1)
    
    # --- RUN THE ASYNC FUNCTION ---
    try:
        asyncio.run(run_cli_async()) # <--- Use asyncio.run()
    except KeyboardInterrupt:
        log.info("Main async loop interrupted by user.")
    except Exception as main_e:
        log.exception(f"Fatal error in main async execution: {main_e}")

# --- END OF FILE C:\Rix_Dev\Pro_Rix\run_rix_cli.py ---