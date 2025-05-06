# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\orchestration\local_pipeline.py ---
# Version: 1.1.0 (V52.1 - Implemented Pipeline Logic)

import logging
import json
import time
import datetime
from typing import Dict, Any, Optional, List, Tuple, Literal
import readline # Optional: for better input editing in terminal
# --- Rix Core Module Imports (V52 Structure) ---
try:
    from Rix_Brain.core import global_state as RixState
    from Rix_Brain.core import config_manager as rix_config
    from Rix_Brain.core import utils as rix_utils
    from Rix_Brain.core.exceptions import LLMInvocationError, ToolExecutionError, OrchestrationError
    from Rix_Brain.services import vertex_llm as rix_llm_calls
    from Rix_Brain.services import firestore_history as rix_history_manager
    from Rix_Brain.services import pgvector_memory as rix_vector_pg
    from Rix_Brain.tools import tool_manager as Rix_tools_play
    print(f"--- {__name__}: Imports successful.", flush=True)
except ImportError as e:
    print(f"--- {__name__} FATAL: Import Error: {e}. Check structure/paths.", flush=True)
    raise
except Exception as e:
    print(f"--- {__name__} FATAL: Unexpected Import Error: {e}.", flush=True)
    raise

# --- Logging ---
log = logging.getLogger("Rix_Heart")

# --- Constants ---
try: MAX_ORCHESTRATION_STEPS = int(rix_config.get_config("MAX_ORCHESTRATION_STEPS", 15))
except Exception: MAX_ORCHESTRATION_STEPS = 15

# --- SocketIO Handling (Conditional) ---
_socketio_instance = None; _is_web_mode = False
log.info("Orchestrator: Initialized in LOCAL MODE (No direct SocketIO import).")

def _conditional_emit(event_name: str, data: Dict, room_sid: str):
    # ... (Keep implementation from V1.1.0) ...
    if _socketio_instance and _is_web_mode:
        try: _socketio_instance.emit(event_name, data, room=room_sid); log.debug(f"[{room_sid}] Emitted '{event_name}' via SocketIO.")
        except Exception as e: log.error(f"[{room_sid}] SocketIO emit failed '{event_name}': {e}", exc_info=False)
    else: log.info(f"[{room_sid}] LOCAL MODE: Skip Emit '{event_name}', Data='{str(data)[:100]}...'")


# --- Helper: _save_and_emit ---
def _save_and_emit(session_id: str, message_dict: Dict[str, Any], current_history: List[Dict]) -> Tuple[bool, Optional[int], Optional[Dict]]:
    # ... (Keep implementation from V1.1.0) ...
    if not rix_history_manager: log.error(f"[{session_id}] History Manager missing."); return False, None, None
    content = message_dict.get('content')
    if not isinstance(content, (str, int, float, bool, list, dict, type(None))):
        log.warning(f"[{session_id}] Converting non-standard content type '{type(content).__name__}' to string.")
        message_dict['content'] = str(content)
    success, index = rix_history_manager.append_message(session_id, message_dict)
    msg_with_idx = None
    if success and index is not None:
        msg_with_idx = message_dict.copy(); msg_with_idx["index"] = index
        if isinstance(current_history, list): current_history.append(msg_with_idx)
        else: log.error(f"[{session_id}] _save_and_emit: current_history not list!")
        log.info(f"[{session_id}] Saved Idx={index}, Actor={message_dict.get('actor')}, Target={message_dict.get('target_recipient')}")
        _conditional_emit("history_step_added", {"session_id": session_id, "history_entry": msg_with_idx}, room_sid=session_id)
    else: log.error(f"[{session_id}] Save history fail! Actor={message_dict.get('actor')}")
    return success, index, msg_with_idx

# --- Main Orchestration Pipeline ---

def run_agent_pipeline(session_id: str, initial_user_message: str) -> Dict:
    """
    Runs the local, synchronous Rix agent pipeline (V52.1 Implementation).
    """
    log_prefix = f"[{session_id}] OrchPipe"
    log.info(f"{log_prefix} V1.1.0: Starting local synchronous pipeline...")
    start_time = time.monotonic()

    # --- Verify Initialization ---
    if not RixState or not RixState.system_initialized:
        init_error = getattr(RixState, 'initialization_error', 'Unknown Init Error')
        log.error(f"{log_prefix} Rix Brain not initialized. Aborting. Error: {init_error}")
        return {"status": "failed", "error": f"Init Error: {init_error}", "full_history": []}
    if not all([rix_history_manager, rix_llm_calls, rix_utils, Rix_tools_play]):
        log.error(f"{log_prefix} Core Rix modules missing. Aborting.")
        return {"status": "failed", "error": "Core Rix modules unavailable.", "full_history": []}

    # --- Pipeline State ---
    current_history: List[Dict[str, Any]] = []
    error_details: Optional[str] = None
    step_count = 0
    final_response = "[Pipeline Error: No final response determined]" # Default Error
    last_message: Optional[Dict] = None # Initialize last_message

    try:
        # --- Step 0: User Input ---
        log.info(f"{log_prefix} Step 0: Processing initial user message.")
        user_entry = rix_history_manager.create_history_entry_dict(
            actor="User", content=initial_user_message, message_type="user_input", target_recipient="@Classifier"
        )
        ok, idx, saved_msg = _save_and_emit(session_id, user_entry, current_history)
        if not ok or saved_msg is None: raise OrchestrationError("Failed to save initial user message.", step=0)
        last_message = saved_msg
        step_count += 1

        # --- Main Agent Loop ---
        while step_count <= MAX_ORCHESTRATION_STEPS:
            log.info(f"{log_prefix} --- Step {step_count}/{MAX_ORCHESTRATION_STEPS} ---")
            if not last_message: raise OrchestrationError("Internal state error: last_message is None.", step=step_count)

            current_target = last_message.get("target_recipient")
            last_actor = last_message.get("actor")
            last_content = last_message.get("content")
            last_index = last_message.get("index", -1)

            log.info(f"{log_prefix} Input: Target='{current_target}', From='{last_actor}', Idx={last_index}")
            log.debug(f"{log_prefix} Input Content Preview: {str(last_content)[:150]}...")

            # --- Termination Conditions ---
            if not current_target: log.info(f"{log_prefix} No target. Ending pipeline naturally."); final_response = "[Pipeline completed: No next target]"; break
            if current_target == "@User": log.info(f"{log_prefix} Target is @User. Ending pipeline."); final_response = last_content; break

            # --- Agent/Tool Invocation Logic ---
            next_actor: Optional[str] = None; next_content: Any = None
            next_target: Optional[str] = None; next_msg_type: str = "internal_status"
            history_entry_extras: Dict = {} # For storing structured data

            try:
                # --- @Classifier Logic ---
                if current_target == "@Classifier":
                    next_actor = "Classifier"
                    log.info(f"{log_prefix} Calling Classifier Agent...")
                    classifier_output = rix_llm_calls.invoke_classifier_agent(session_id, last_message, current_history)
                    if not classifier_output: raise LLMInvocationError("Classifier returned None.", role="Classifier")
                    next_content = json.dumps(classifier_output) # Store raw JSON output

                    if classifier_output.get("action") == "recall_memory":
                        next_msg_type = "memory_recall_guidance"
                        log.info(f"{log_prefix} Classifier guided Memory Recall.")
                        requesting_actor_target = "@Manager" # Default
                        for msg in reversed(current_history[:-1]):
                            if msg.get("message_type") == "internal_question" and msg.get("target_recipient") == "@Classifier":
                                 actor_name = msg.get("actor", "Manager"); requesting_actor_target = f"@{actor_name}"; break
                        next_target = requesting_actor_target
                        log.info(f"{log_prefix} Executing memory recall for {next_target}...")
                        recall_params = classifier_output
                        recalled_data = rix_vector_pg.recall_memory(
                            query_text=recall_params.get("query_text", ""), top_n=recall_params.get("top_n", 3), filter_category=recall_params.get("filter_category")
                        )
                        next_actor = "MemorySystem"; next_content = {"recalled_memories": recalled_data}
                        next_msg_type = "memory_recall_result"
                        log.info(f"{log_prefix} Recall found {len(recalled_data)} items.")
                    elif "classification" in classifier_output:
                        classification = classifier_output.get("classification", "UNKNOWN").upper()
                        next_msg_type = "classification_result"
                        log.info(f"{log_prefix} Classifier result: {classification}")
                        next_target = "@Manager" # Always route to Manager
                        if classification not in ["CHAT", "ASK", "WORK"]: log.warning(f"{log_prefix} Unknown classification '{classification}'.")
                    else: raise ValueError(f"Classifier output unknown format: {classifier_output}")

                # --- @Manager Logic ---
                elif current_target == "@Manager":
                    next_actor = "Manager"
                    log.info(f"{log_prefix} Calling Manager Agent...")
                    last_msg_type = last_message.get("message_type")
                    log.debug(f"{log_prefix} Manager received input type: {last_msg_type}")

                    # Default to finalizing if unsure, using last content as input
                    final_input = str(last_content)
                    user_msg = next((m['content'] for m in reversed(current_history) if m['actor']=='User'), "[Context Lost]")

                    if last_msg_type == "classification_result":
                        classification = json.loads(last_content or '{}').get("classification", "UNKNOWN").upper()
                        if classification == "CHAT":
                            log.info(f"{log_prefix} Manager: Handling CHAT..."); final_input = f"[Direct CHAT]: {user_msg}"; next_msg_type = "system_response"; next_target = "@User"
                        elif classification == "ASK":
                            log.info(f"{log_prefix} Manager: Handling ASK - Requesting memory..."); next_content = f"Need context/memory for: '{user_msg[:100]}...'"; next_target = "@Classifier"; next_msg_type = "internal_question"
                        elif classification == "WORK":
                            log.info(f"{log_prefix} Manager: Handling WORK - Dispatching..."); next_content, next_target = rix_llm_calls.invoke_manager_dispatch(session_id, user_msg, current_history); next_msg_type = "dispatch_instruction"
                        else: log.warning(f"{log_prefix} Manager: Unknown classification '{classification}'. Finalizing."); final_input = f"[Fallback Request]: {user_msg}"; next_msg_type = "system_response"; next_target = "@User"
                    elif last_msg_type == "tool_action_result": log.info(f"{log_prefix} Manager: Processing Tool Result..."); final_input = f"[Tool Result Received]: {str(last_content)[:500]}..."; next_msg_type = "system_response"; next_target = "@User"
                    elif last_msg_type in ["internal_report", "internal_status"]: log.info(f"{log_prefix} Manager: Processing Thinker Report/Status..."); final_input = str(last_content); next_msg_type = "system_response"; next_target = "@User"
                    elif last_msg_type == "memory_recall_result": log.info(f"{log_prefix} Manager: Processing Memory Result for ASK..."); final_input = f"[Direct ASK Request]: User asked '{user_msg}'. Memory Context: {str(last_content)[:500]}..."; next_msg_type = "system_response"; next_target = "@User"
                    else: log.warning(f"{log_prefix} Manager: Unhandled type '{last_msg_type}'. Finalizing."); final_input = f"[Unexpected Input]: {str(last_content)[:200]}..."; next_msg_type = "system_response"; next_target = "@User"

                    # If we determined a next step other than finalize, set content/target
                    if next_target != "@User": pass # Content/Target already set above for non-finalize cases
                    else: # Otherwise, call finalize
                         log.debug(f"{log_prefix} Calling manager finalize with input: {final_input[:100]}...")
                         next_content, final_target = rix_llm_calls.invoke_manager_finalize(session_id, final_input, current_history)
                         next_target = final_target or "@User" # Ensure target is User

                # --- @Thinker Logic ---
                elif current_target == "@Thinker":
                    next_actor = "Thinker"
                    log.info(f"{log_prefix} Calling Thinker Agent...")
                    tool_schemas = Rix_tools_play.get_tool_schemas()

                    if last_message.get("message_type") == "tool_action_result":
                        log.info(f"{log_prefix} Thinker: Processing Tool Result...")
                        thinker_output_text, tool_calls_requested = rix_llm_calls.invoke_model_with_function_response(session_id, current_history)
                    else: # Assume dispatch instruction
                        log.info(f"{log_prefix} Thinker: Processing Dispatch Instruction...")
                        thinker_output_text, tool_calls_requested = rix_llm_calls.invoke_thinker_plan(session_id, str(last_content), current_history, tool_schemas)

                    if tool_calls_requested:
                        log.info(f"{log_prefix} Thinker: Requesting tool call(s).")
                        next_content = f"[Requesting {len(tool_calls_requested)} tool call(s)...]" # Placeholder content
                        next_target = "@ToolExecutor"; next_msg_type = "tool_action_request"
                        history_entry_extras["function_call_data"] = tool_calls_requested
                        if tool_calls_requested: # Add first tool info for convenience
                             history_entry_extras["tool_name"] = tool_calls_requested[0].get('name')
                             history_entry_extras["tool_args"] = tool_calls_requested[0].get('args')
                    elif thinker_output_text:
                        log.info(f"{log_prefix} Thinker: Providing text report/question.")
                        next_content = thinker_output_text
                        if not next_content.startswith("@Manager:"): log.warning(f"{log_prefix} Thinker text missing '@Manager:'."); next_content = f"@Manager: {next_content}"
                        next_target = "@Manager"; next_msg_type = "internal_report"
                    else: raise LLMInvocationError("Thinker failed to produce output.", role="Thinker")

                # --- @ToolExecutor Logic ---
                elif current_target == "@ToolExecutor":
                    next_actor = "Tool"
                    log.info(f"{log_prefix} Calling Tool Executor...")
                    tool_request_list = last_message.get("function_call_data") # Get from structured data first
                    if not isinstance(tool_request_list, list): tool_request_list = [] # Ensure list
                    if not tool_request_list: # Fallback parsing
                        try: parsed_content = json.loads(last_content or '[]'); tool_request_list = [parsed_content] if isinstance(parsed_content, dict) else parsed_content if isinstance(parsed_content, list) else []
                        except: raise OrchestrationError(f"Invalid tool request data. Content: {last_content}", step=step_count)
                    if not tool_request_list: raise OrchestrationError("No tool requests found.", step=step_count)

                    # --- Execute ONLY FIRST Tool Call ---
                    if len(tool_request_list) > 1: log.warning(f"{log_prefix} Executing only FIRST of {len(tool_request_list)} tool calls.")
                    tool_call = tool_request_list[0]
                    tool_name = tool_call.get("name"); tool_args = tool_call.get("args", {})
                    if not tool_name or not isinstance(tool_args, dict): raise ValueError(f"Invalid tool call format: {tool_call}")

                    log.info(f"{log_prefix} Executing Tool: '{tool_name}' Args: {tool_args}")
                    tool_result = {}
                    try:
                        # --- MOCKING ---
                        if session_id.startswith("cli_session_mock"): # Simple flag using session id
                            log.warning(f"{log_prefix} MOCKING tool '{tool_name}' execution.")
                            if tool_name == "list_files": tool_result = {"command": tool_name, "status": "success", "result": {"files": ["mock_soul1.json", "mock_soul2.json"], "directories": ["core", "services"]}, "message":"Mock success"}
                            elif tool_name == "read_file": tool_result = {"command": tool_name, "status": "success", "result": "[Mocked content of file '{}']".format(tool_args.get('path')), "message":"Mock success"}
                            else: tool_result = {"command": tool_name, "status": "failed", "error": {"code": "ERR_MOCK", "message": "Tool not mocked."}, "message": "Tool not mocked"}
                        # --- REAL EXECUTION ---
                        else:
                             tool_result = Rix_tools_play.execute_tool(tool_name, tool_args, session_id)
                    except Exception as tool_exec_e:
                         log.exception(f"{log_prefix} Tool execution failed with exception: {tool_exec_e}")
                         tool_result = {"command": tool_name, "status": "failed", "error": {"code": "ERR_TOOL_EXCEPTION", "message": str(tool_exec_e)},"metadata": {"input_args": tool_args}}

                    log.info(f"{log_prefix} Tool '{tool_name}' finished. Status: {tool_result.get('status')}")
                    next_msg_type = "tool_action_result"; next_target = "@Thinker" # Send result back to Thinker
                    history_entry_extras["tool_name"] = tool_name
                    history_entry_extras["tool_status"] = tool_result.get("status")
                    function_response_content = { k: v for k, v in tool_result.items() if k in ['status', 'result', 'message', 'error'] and v is not None}
                    history_entry_extras["function_response_data"] = {"name": tool_name, "response": function_response_content}
                    next_content = json.dumps(function_response_content) # Content is inner response dict

                # --- @MemoryWriter (Placeholder Handling) ---
                elif current_target == "@MemoryWriter":
                    log.warning(f"{log_prefix} Received target @MemoryWriter - Placeholder. Ending turn.")
                    next_actor = "System"; next_content = "[Memory Writer Skipped]"; next_target = None; next_msg_type = "internal_status"

                # --- Unhandled Target ---
                else: raise OrchestrationError(f"Unhandled target recipient: {current_target}", step=step_count)

            except Exception as agent_err: # Catch errors during agent logic/calls
                log.exception(f"{log_prefix} Error processing target '{current_target}': {agent_err}")
                error_details = f"Error in step {step_count} ({current_target}): {agent_err}"
                next_actor = "System"; next_content = f"Pipeline Error: {error_details[:500]}"
                next_target = None; next_msg_type = "system_error"
                # Fall through to save the error message

            # --- Save Step Result ---
            if next_actor and next_content is not None:
                 history_entry_data = {
                    "actor": next_actor, "content": next_content, "message_type": next_msg_type,
                    "target_recipient": next_target, **history_entry_extras # Merge extras
                 }
                 new_entry = rix_history_manager.create_history_entry_dict(**history_entry_data)
                 ok, idx, saved_msg = _save_and_emit(session_id, new_entry, current_history)
                 if not ok: error_details = error_details or f"CRITICAL: Failed history save step {step_count+1}!"; log.critical(error_details); break
                 last_message = saved_msg
                 if error_details or not next_target: break # Exit loop if error occurred or no next target
            else: # Should not happen if logic is correct, implies agent failed to produce output
                 log.error(f"{log_prefix} Agent for target '{current_target}' failed to produce output. Halting.")
                 error_details = error_details or f"Agent {current_target} produced no output."
                 break

            step_count += 1 # Increment step count AFTER successful processing

        # --- End of While Loop ---

        if error_details: log.error(f"{log_prefix} Pipeline terminated early: {error_details}"); final_response = f"[Pipeline Error: {error_details}]"
        elif step_count > MAX_ORCHESTRATION_STEPS: log.warning(f"{log_prefix} Max steps reached."); error_details = "Max orchestration steps reached."; final_response = "[Pipeline Error: Max steps reached]"
        elif not last_message or last_message.get('target_recipient') is None: log.info(f"{log_prefix} Pipeline ended naturally (No target)."); final_response = "[Pipeline completed: No further action]"
        # else: target was @User, final_response already set from last_message.content

    # --- Catch Outer Errors ---
    except OrchestrationError as oe: log.exception(f"{log_prefix} Orchestration Error: {oe}"); error_details = str(oe); final_response = f"[Orchestration Error: {oe}]"
    except LLMInvocationError as llm_err: log.error(f"{log_prefix} Pipeline LLM Error: {llm_err}"); error_details = str(llm_err); final_response = f"[LLM Error: {llm_err}]"
    except Exception as e: log.exception(f"{log_prefix} Unexpected Pipeline Error: {e}"); error_details = f"Unexpected Error: {e}"; final_response = f"[Unexpected Error: {e}]"

    # --- Final Processing ---
    finally:
        duration = time.monotonic() - start_time
        final_status = "failed" if error_details else "success"
        log.info(f"{log_prefix} Pipeline finished. Status: {final_status}. Duration: {duration:.3f}s. Steps: {step_count}.")
        if error_details: log.error(f"{log_prefix} Final Error: {error_details}")
        log.info(f"{log_prefix} Final User Response: {str(final_response)[:150]}...")

        # --- Memory Writing ---
        if RixState and RixState.system_initialized and rix_llm_calls:
             log.info(f"{log_prefix} Triggering final memory writer...")
             flow_t = "ERROR" if error_details else "UNKNOWN"
             class_msg = next((m for m in current_history if m.get('message_type') == 'classification_result'), None)
             if class_msg and isinstance(class_msg.get('content'), str):
                  try: classification = json.loads(class_msg['content']).get("classification", "UNKNOWN").upper()
                  except: pass
                  if classification == "CHAT": flow_t = "CHAT"
                  elif classification == "ASK": flow_t = "ASK"
                  elif classification == "WORK": flow_t = "WORK"
             flow_t = "ERROR" if error_details else flow_t # Error overrides classification guess
             try:
                 memory_snippet = current_history[-6:]
                 summary = rix_llm_calls.invoke_memory_writer(session_id, memory_snippet, final_status, flow_type=flow_t)
                 log.info(f"{log_prefix} Memory summary generated (sample): {summary[:100]}...")
             except Exception as mem_e: log.error(f"{log_prefix} Error during final memory write call: {mem_e}", exc_info=True)

        return {"status": final_status, "error": error_details, "final_response": final_response, "full_history": current_history}

# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\orchestration\local_pipeline.py ---