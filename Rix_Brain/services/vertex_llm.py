# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\services\vertex_llm.py ---
# Version: 1.9.1 (V52.0 - Updated Imports & Full Logic) - Formerly rix_llm_calls.py
# Author: Vishal Sharma / Gemini
# Date: May 6, 2025
# Description: Handles LLM calls using Vertex AI SDK for Rix V52 Local Orchestration.

print(f"--- Loading Module: {__name__} (V1.9.1 - V52 Imports) ---", flush=True)

import logging
import json
import time
import sys
import os
import re
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Literal

# --- Rix Core Module Imports (V52 Structure) ---
try:
    # Core
    from Rix_Brain.core import config_manager as rix_config # Alias
    from Rix_Brain.core import utils as rix_utils # Alias
    from Rix_Brain.core import global_state as RixState # Alias
    from Rix_Brain.core.exceptions import LLMInvocationError # Direct import
    # Services (needed for memory context)
    from Rix_Brain.services import pgvector_memory as rix_vector_pg # Alias
    # Tools (needed for schema context)
    from Rix_Brain.tools import tool_manager as Rix_tools_play # Alias

    # Need SCIIC for context helper - assuming integrated for now
    # from Rix_Brain.??? import Rix_SCIIC_cycle_manager # Where does this live now? Let's keep _get_context_for_llm simplified for now.

    RIX_BRAIN_LOADED = True # Assume if these import, the core is available
    print(f"--- {__name__}: Rix Brain core imports successful.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Import Fail: {e}", flush=True)
    RIX_BRAIN_LOADED = False
    # Define dummy exception if core failed
    class LLMInvocationError(Exception): pass
    # Set modules to None
    rix_config, rix_utils, RixState, Rix_tools_play, rix_vector_pg = None, None, None, None, None


# --- Import Google Cloud / Vertex AI Libraries ---
VERTEX_AI_AVAILABLE = False
# Define dummy classes first
VertexGenerativeModel, VertexContent, VertexPart, VertexGenerationConfig = object, object, object, object
VertexHarmCategory, VertexHarmBlockThreshold, VertexTool = object, object, object
FunctionDeclaration, FunctionCall, FunctionResponse = object, object, object
vertex_exceptions, TextEmbeddingModel = object, object

try:
    print(f"--- {__name__}: Attempting Vertex AI SDK Imports...", flush=True)
    import google.cloud.aiplatform
    import vertexai.preview.generative_models
    import vertexai.language_models
    import google.api_core.exceptions

    import vertexai
    from vertexai.preview.generative_models import (
        GenerativeModel as VertexGenerativeModel,
        Content as VertexContent, Part as VertexPart,
        GenerationConfig as VertexGenerationConfig,
        HarmCategory as VertexHarmCategory,
        HarmBlockThreshold as VertexHarmBlockThreshold,
        Tool as VertexTool, FunctionDeclaration, FunctionCall
    )
    try: from vertexai import FunctionResponse
    except ImportError: pass # Keep dummy if not found

    from google.api_core import exceptions as vertex_exceptions
    from vertexai.language_models import TextEmbeddingModel

    VERTEX_AI_AVAILABLE = True
    logger = logging.getLogger("Rix_Heart")
    logger.info("Vertex LLM Service: Vertex AI SDK core components imported successfully.")
    print(f"--- {__name__}: Vertex AI SDK Import SUCCESS.", flush=True)
except ImportError as e:
    logger = logging.getLogger("Rix_Heart")
    logger.error(f"{__name__}: Vertex AI SDK Import Error: {e}. LLM calls will fail.", exc_info=False)
    print(f"--- {__name__}: Vertex AI SDK Import FAILED: {e}", flush=True)
    VERTEX_AI_AVAILABLE = False
except Exception as import_err:
    logger = logging.getLogger("Rix_Heart")
    logger.exception(f"{__name__}: Unexpected Vertex AI import error: {import_err}", exc_info=True)
    print(f"--- {__name__}: Unexpected Vertex AI Import Error: {import_err}", flush=True)
    VERTEX_AI_AVAILABLE = False

# --- Logging Setup ---
logger = logging.getLogger("Rix_Heart")

# --- Helper Functions ---

def _get_souls_dir() -> Optional[Path]:
    """Gets the directory path for agent soul prompts from Rix config."""
    if not RIX_BRAIN_LOADED or not rix_config:
        logger.error(f"{__name__}: Cannot get souls dir: Rix config not loaded.")
        return None
    try:
        # Attempt to find Rix_Brain relative to this file if config doesn't have paths
        try:
            current_dir = Path(__file__).parent.resolve() # Rix_Brain/services
            brain_dir = current_dir.parent # Rix_Brain
        except NameError:
             brain_dir = Path("Rix_Brain").resolve() # Fallback

        # Use the new path structure
        souls_dir = brain_dir / "agents" / "souls"
        if souls_dir.is_dir():
            logger.debug(f"{__name__}: Found souls directory: {souls_dir}")
            return souls_dir
        else:
            logger.error(f"{__name__}: Souls directory not found or invalid: {souls_dir}")
            return None
    except Exception as e:
        logger.exception(f"{__name__}: Error getting souls dir path: {e}")
        return None

# --- History Conversion Helper ---
def _convert_history_to_vertex_content(messages: List[Dict]) -> List[VertexContent]:
    """
    Converts Rix history message dictionaries into the Vertex AI Content format,
    correctly handling text, function calls (model output), and function
    responses (tool output).
    Version: 1.9.1 (Based on V1.8.4 logic)
    """
    if not VERTEX_AI_AVAILABLE: logger.error("Vertex AI types missing for history conversion."); return []
    vertex_messages: List[VertexContent] = []; logger.debug(f"Converting {len(messages)} history entries (V1.9.1)...")
    for i, msg_dict in enumerate(messages):
        role = None; vertex_parts = []; is_fn_call = False; is_fn_resp = False
        fn_call_data = msg_dict.get("function_call_data")
        fn_resp_data = msg_dict.get("function_response_data")
        text_content = msg_dict.get("content"); actor = msg_dict.get("actor"); name = msg_dict.get("name"); msg_type = msg_dict.get("message_type")

        try:
            if fn_call_data and isinstance(fn_call_data, list):
                role = "model"; is_fn_call = True
                for call in fn_call_data:
                    if isinstance(call, dict) and "name" in call and "args" in call:
                         args_d = dict(call["args"]) if hasattr(call["args"], "items") else {}
                         vertex_parts.append(VertexPart.from_function_call(FunctionCall(name=call["name"], args=args_d)))
                    else: logger.warning(f"Msg {i} Invalid FnCall data in list: {call}")
                if text_content and isinstance(text_content, str) and text_content.strip(): vertex_parts.append(VertexPart.from_text(text_content.strip()))

            elif fn_resp_data and isinstance(fn_resp_data, dict):
                 tool_name = fn_resp_data.get("name")
                 resp_content_dict = fn_resp_data.get("response")
                 if tool_name and resp_content_dict is not None:
                     role = "function"; is_fn_resp = True
                     vertex_resp = resp_content_dict if isinstance(resp_content_dict, dict) else {"result": str(resp_content_dict)}
                     if FunctionResponse is not object: # Check if imported
                          vertex_parts.append(VertexPart.from_function_response(FunctionResponse(name=tool_name, response=vertex_resp)))
                     else:
                          logger.error(f"Msg {i}: FunctionResponse type unavailable, cannot create part for tool '{tool_name}'.")
                          vertex_parts.append(VertexPart.from_text(f"[Function Response Error: Type Missing] Tool: {tool_name}, Response: {str(vertex_resp)[:50]}..."))
                 else: logger.warning(f"Msg {i} Invalid FnResp data format: {fn_resp_data}")
                 if text_content and isinstance(text_content, str) and text_content.strip(): vertex_parts.append(VertexPart.from_text(text_content.strip()))

            elif msg_type == "tool_action_result" and isinstance(text_content, str):
                try:
                    tool_result_dict = json.loads(text_content)
                    tool_name = tool_result_dict.get("command") or tool_result_dict.get("tool_name")
                    response_content_for_llm = { k: v for k, v in tool_result_dict.items() if k in ['status', 'result', 'message', 'error'] and v is not None}
                    if tool_name:
                        role = "function"; is_fn_resp = True
                        if FunctionResponse is not object: # Check if imported
                             vertex_parts.append(VertexPart.from_function_response(FunctionResponse(name=tool_name, response=response_content_for_llm)))
                             logger.debug(f"Msg {i} Converted tool_action_result text to FunctionResponse.")
                        else:
                              logger.error(f"Msg {i}: FunctionResponse type unavailable, cannot create part for tool '{tool_name}'.")
                              vertex_parts.append(VertexPart.from_text(f"[Function Response Error: Type Missing] Tool: {tool_name}, Response: {str(response_content_for_llm)[:50]}..."))
                    else: raise ValueError("Tool name missing in parsed result")
                except Exception as e:
                    logger.warning(f"Msg {i} Failed parsing tool_action_result text: {e}. Treating as plain text.")
                    role = "model"; vertex_parts.append(VertexPart.from_text(f"[Tool Result (Error Parsing)]: {text_content}"))

            elif text_content is not None: # Regular Text Message
                content_str = str(text_content).strip()
                if not content_str: continue
                if actor == "User" or msg_type == "user_input": role = "user"
                elif actor in ["Manager", "Thinker", "System", "Classifier", "MemoryWriter", "FinalResponseGen", "DirectAnswerService", "MemorySystem"] or msg_type in ["AIMessage", "SystemMessage", "internal_status", "internal_question", "internal_answer", "system_response", "internal_report", "memory_recall_result", "classification_result", "memory_recall_guidance"]: role = "model"
                elif actor == "Tool" and msg_type != "tool_action_result": role = "model"; content_str = f"[Tool Note ({name})]: {content_str}"
                else: logger.warning(f"Msg {i}: Unknown actor/type '{actor}'/'{msg_type}'. Defaulting role to 'model'."); role = "model"
                vertex_parts.append(VertexPart.from_text(content_str))

            if role and vertex_parts:
                can_merge = (vertex_messages and vertex_messages[-1].role == role and not is_fn_call and not is_fn_resp and
                             not (vertex_messages[-1].parts and hasattr(vertex_messages[-1].parts[0], 'function_call')) and
                             not (vertex_messages[-1].parts and hasattr(vertex_messages[-1].parts[0], 'function_response')))
                if can_merge: vertex_messages[-1].parts.extend(vertex_parts)
                else: vertex_messages.append(VertexContent(role=role, parts=vertex_parts))
            elif text_content or fn_call_data or fn_resp_data:
                 logger.warning(f"Msg {i}: No role/parts for message. Skipping. Data: {str(msg_dict)[:100]}...")
        except Exception as conv_e: logger.error(f"Msg {i} Error during history conversion: {conv_e}", exc_info=True); continue

    if not vertex_messages or vertex_messages[0].role not in ["user", "model"]:
        logger.warning(f"{__name__}: History doesn't start with user/model. Prepending dummy."); vertex_messages.insert(0, VertexContent(role="user", parts=[VertexPart.from_text("<Start Conversation>")]))
    final_vertex_messages = [msg for msg in vertex_messages if msg.parts]
    logger.debug(f"{__name__}: Converted {len(messages)} -> {len(final_vertex_messages)} Vertex Content objects.")
    return final_vertex_messages

# --- LLM Invocation Helper ---
def _invoke_llm_with_retry(
    model_name: str,
    vertex_formatted_contents: List[VertexContent],
    temperature: float,
    session_id: str,
    invocation_role: str,
    tools: Optional[List[VertexTool]] = None,
    max_retries: int = 1,
    initial_delay: float = 2.0,
    max_delay: float = 10.0
) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    Invokes Vertex AI LLM with retry logic. Handles text and native function calls.
    Version: 1.9.1 (Based on V1.8.4 logic)
    """
    if not VERTEX_AI_AVAILABLE: raise LLMInvocationError("Vertex AI SDK missing.", role=invocation_role)
    if not all([VertexGenerativeModel, VertexGenerationConfig, VertexHarmCategory, VertexHarmBlockThreshold, VertexContent, VertexPart, FunctionCall]):
         # Don't fail if only FunctionResponse is missing
         logger.warning(f"Required Vertex AI types potentially missing (FunctionResponse was {FunctionResponse})")
         # raise LLMInvocationError("Required Vertex AI types missing.", role=invocation_role) # Don't raise here yet

    if not vertex_formatted_contents: logger.warning(f"[{session_id}] {invocation_role}: Invoking with empty message list.")

    delay = initial_delay; last_exception = None
    try:
        safety_settings = {vhc: VertexHarmBlockThreshold.BLOCK_NONE for vhc in VertexHarmCategory}
        temp_float = max(0.0, min(float(temperature), 2.0))
        generation_config = VertexGenerationConfig(temperature=temp_float)
        logger.debug(f"[{session_id}] {invocation_role}: Vertex Config: Temp={temp_float}, Safety=BLOCK_NONE")
    except Exception as cfg_e: raise LLMInvocationError("Vertex generation config failed.", role=invocation_role, original_exception=cfg_e)

    for attempt in range(max_retries + 1):
        try:
            log_prefix = f"[{session_id}] {invocation_role}({model_name})"
            logger.info(f"{log_prefix}: Invoking LLM (Attempt {attempt + 1}/{max_retries + 1}). Tools: {'Yes' if tools else 'No'}")
            logger.debug(f"{log_prefix}: History length: {len(vertex_formatted_contents)} Content objects.")
            # Ensure model name has 'models/' prefix if needed by SDK version
            if not model_name.startswith("models/"):
                 resolved_model_name = f"models/{model_name}"
                 logger.debug(f"Prepending 'models/' prefix: {resolved_model_name}")
            else:
                 resolved_model_name = model_name

            model = VertexGenerativeModel(resolved_model_name)
            response = model.generate_content(
                contents=vertex_formatted_contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools
            )
            logger.debug(f"{log_prefix}: Raw LLM response received.")

            resp_txt: Optional[str] = None; fn_calls_out: Optional[List[Dict]] = None
            block_reason: Optional[str] = None; finish_reason: str = "UNKNOWN"

            if hasattr(response, "prompt_feedback") and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_enum = response.prompt_feedback.block_reason; block_reason = getattr(block_enum, "name", str(block_enum))
                logger.error(f"{log_prefix}: Prompt Blocked! Reason: {block_reason}.")
                raise LLMInvocationError(f"Vertex prompt blocked ({block_reason})", role=invocation_role)

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"): finish_enum = candidate.finish_reason; finish_reason = getattr(finish_enum, "name", str(finish_enum))
                if finish_reason == "SAFETY": logger.error(f"{log_prefix}: Response Blocked by SAFETY filter."); raise LLMInvocationError("Vertex response blocked: SAFETY.", role=invocation_role)

                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    all_parts = candidate.content.parts
                    fn_call_parts_found = [p for p in all_parts if hasattr(p, "function_call") and p.function_call]
                    if fn_call_parts_found:
                        logger.info(f"{log_prefix}: Detected {len(fn_call_parts_found)} function call(s). Finish: {finish_reason}")
                        fn_calls_list = []
                        for part in fn_call_parts_found:
                            fc = part.function_call
                            args_d = dict(fc.args) if hasattr(fc.args, "items") else {}
                            fn_calls_list.append({"name": fc.name, "args": args_d})
                        if fn_calls_list:
                            return None, fn_calls_list # SUCCESS - Function Call(s)
                        else: logger.error(f"{log_prefix}: Found fn_call parts but failed to parse.")
                    else:
                        text_parts_content = [p.text for p in all_parts if hasattr(p, "text") and p.text is not None]
                        if text_parts_content:
                            resp_txt = "".join(text_parts_content).strip()
                            logger.info(f"{log_prefix}: Received text response (Len: {len(resp_txt)}). Finish: {finish_reason}")
                            if finish_reason != "STOP": logger.warning(f"{log_prefix}: Non-STOP finish: {finish_reason}.")
                            return resp_txt, None # SUCCESS - Text Response
                        else: last_exception = ValueError(f"LLM parts but no text/fn_calls (Finish: {finish_reason}). Parts: {all_parts}")
                else: last_exception = ValueError(f"LLM candidate empty/missing parts (Finish: {finish_reason}).")
            else: last_exception = ValueError("LLM response has no candidates."); raise LLMInvocationError(str(last_exception), role=invocation_role)

        except (vertex_exceptions.ResourceExhausted, vertex_exceptions.ServiceUnavailable) as e: last_exception = e; logger.warning(f"{log_prefix}: Retryable API Error (Attempt {attempt + 1}): {e}")
        except LLMInvocationError as e: logger.error(f"{log_prefix}: Explicit LLMInvocationError (Attempt {attempt + 1}): {e}"); raise e
        except Exception as e: last_exception = e; logger.warning(f"{log_prefix}: Unexpected LLM invoke error (Attempt {attempt + 1}): {e}", exc_info=logger.isEnabledFor(logging.DEBUG))

        if last_exception and attempt < max_retries: logger.info(f"{log_prefix}: Error, pausing {delay:.1f}s before retry {attempt + 2}..."); time.sleep(delay); delay = min(delay * 2, max_delay); continue
        elif last_exception: logger.error(f"{log_prefix}: LLM call failed after {max_retries + 1} attempts."); raise LLMInvocationError(f"LLM call failed after retries: {last_exception}", role=invocation_role, original_exception=last_exception)
        break

    logger.critical(f"{log_prefix}: LLM invocation loop completed unexpectedly."); raise LLMInvocationError("LLM invocation loop exited unexpectedly.", role=invocation_role, original_exception=last_exception)


# --- Context Retrieval Helper (Simplified for V52 Local) ---
def _get_context_for_llm(
    session_id: str,
    current_goal: str,
    history_context: List[Dict],
    role: Literal["Manager", "Thinker"]
) -> Dict[str, Any]:
    """Gets simplified context (history snippet, memories) for Manager or Thinker."""
    log_prefix = f"[{session_id}] _get_context(for {role})"
    # Use aliases for imported modules from V52 structure
    hist_formatter_available = RIX_BRAIN_LOADED and rix_utils and callable(getattr(rix_utils, 'format_history_snippet', None))
    memory_available = RIX_BRAIN_LOADED and rix_vector_pg and callable(getattr(rix_vector_pg, 'recall_memory', None))

    context_package = {
        "error": None, "recent_history_snippet": [], "relevant_memories": [],
        "personal_memory_hints": {"user_name": "Vishal (Default)", "preferred_style": "Default"},
        "user_goal": current_goal
    }

    # Format History Snippet
    hist_slice = -6 if role == "Thinker" else -4
    hist_list_for_format = history_context[hist_slice:]
    if hist_formatter_available:
        context_package["recent_history_snippet"] = rix_utils.format_history_snippet(hist_list_for_format)
    else:
        context_package["recent_history_snippet"] = ["History Formatter Unavailable"]
        context_package["error"] = (context_package.get("error") or "") + " History Formatter Unavailable."

    # Recall Memories
    if memory_available:
        try:
            query_text = f"{role} context for task: {current_goal}"
            category = 'work_summary' if role == "Thinker" else 'chat_summary'
            top_n = 4 if role == "Thinker" else 3
            logger.debug(f"{log_prefix} Recalling memories (Cat: {category}, N: {top_n})...");
            recalled_data = rix_vector_pg.recall_memory( query_text=query_text, top_n=top_n, filter_category=category )
            if recalled_data:
                formatted_memories = []
                for mem_info in recalled_data:
                    mem_text = mem_info.get('text', '[?]'); dist = mem_info.get('distance', 999); meta = mem_info.get('metadata', {})
                    cat = meta.get('memory_category', '?'); ts = meta.get('timestamp_utc', '?'); status = meta.get('turn_status', '')
                    formatted_memories.append( f"Recalled {cat} ({status} / Dist:{dist:.3f} / TS:{ts}): {mem_text[:120]}..." )
                context_package["relevant_memories"] = formatted_memories
                logger.info(f"{log_prefix} Recalled {len(recalled_data)} memories.")
            else: logger.info(f"{log_prefix} No relevant memories found for query.")
        except Exception as recall_e:
             logger.error(f"{log_prefix} Memory recall error: {recall_e}", exc_info=True)
             context_package["error"] = (context_package.get("error") or "") + f" Recall Error: {recall_e}."
    else:
        context_package["relevant_memories"] = ["Memory Recall Unavailable"]
        context_package["error"] = (context_package.get("error") or "") + " Memory Recall Unavailable."

    ctx_log = { k: (str(v)[:50] + "...") if isinstance(v, (list, dict, str)) and len(str(v)) > 50 else v for k, v in context_package.items() if k != "recent_history_snippet" }
    ctx_log["recent_history_snippet_lines"] = len(context_package["recent_history_snippet"])
    logger.debug(f"{log_prefix} Simplified Context Result: {ctx_log}")
    return context_package


# --- Agent Invocation Functions ---

def invoke_classifier_agent(session_id: str, latest_message_details: Dict[str, Any], history_context: List[Dict]) -> Optional[Dict[str, Any]]:
    """ V51.3: Invokes Classifier Agent for intent classification OR memory recall guidance. """
    invocation_role = "ClassifierAgent"; log_prefix = f"[{session_id}] {invocation_role}"
    logger.info(f"{log_prefix}: Preparing request...")
    if not VERTEX_AI_AVAILABLE: logger.error(f"{log_prefix}: Vertex AI SDK unavailable."); return None
    if not RIX_BRAIN_LOADED or not all([rix_config, rix_utils, RixState]): logger.error(f"{log_prefix}: Rix core missing."); return None
    if not isinstance(latest_message_details, dict) or not all(k in latest_message_details for k in ['actor', 'message_type', 'content']): logger.error(f"{log_prefix}: Invalid 'latest_message_details'."); return None
    try:
        model_name = rix_config.get_config("CLASSIFIER_MODEL")
        temp = float(rix_config.get_config("CLASSIFIER_TEMPERATURE", 0.1))
        sdir = _get_souls_dir(); soul_path = sdir / "classifier_v1_2.json" if sdir else None # Use new soul path
        soul_prompt_template = rix_utils.load_soul_prompt(soul_path) if soul_path else None
        if not model_name or not soul_prompt_template: raise ValueError("Classifier model or soul prompt missing.")
    except Exception as config_e: logger.error(f"{log_prefix}: Error loading config/soul: {config_e}", exc_info=True); return None
    try:
        hist_snippet_list = history_context[-6:]
        hist_str = "\n".join(rix_utils.format_history_snippet(hist_snippet_list)) if rix_utils else "History Unavailable"
        latest_actor = latest_message_details.get("actor", "Unknown")
        latest_type = latest_message_details.get("message_type", "unknown_type")
        latest_target = latest_message_details.get("target_recipient", "Unknown")
        latest_content = str(latest_message_details.get("content", ""))
        final_prompt = soul_prompt_template
        final_prompt = final_prompt.replace("{{RECENT_HISTORY_PLACEHOLDER}}", hist_str or "None available.")
        final_prompt = final_prompt.replace("{{LATEST_MESSAGE_ACTOR_PLACEHOLDER}}", latest_actor)
        final_prompt = final_prompt.replace("{{LATEST_MESSAGE_TYPE_PLACEHOLDER}}", latest_type)
        final_prompt = final_prompt.replace("{{LATEST_MESSAGE_TARGET_PLACEHOLDER}}", latest_target or "None")
        final_prompt = final_prompt.replace("{{LATEST_MESSAGE_CONTENT_PLACEHOLDER}}", latest_content)
        logger.debug(f"{log_prefix} Constructed prompt (Len: {len(final_prompt)}). Latest type: {latest_type}")
    except Exception as prompt_e: logger.error(f"{log_prefix}: Error preparing prompt context: {prompt_e}", exc_info=True); return None
    vertex_content_input = _convert_history_to_vertex_content([{'role': 'user', 'parts': [{'text': final_prompt}]}])
    try:
        resp_txt, _ = _invoke_llm_with_retry(model_name, vertex_content_input, temp, session_id, invocation_role, tools=None, max_retries=1)
    except LLMInvocationError as llm_e: logger.error(f"{log_prefix} LLM Invocation Error: {llm_e}"); return None
    except Exception as e: logger.error(f"{log_prefix} Unexpected LLM Invocation Error: {e}", exc_info=True); return None
    if not resp_txt: logger.error(f"{log_prefix} LLM returned empty response."); return None
    cleaned = resp_txt.strip(); logger.debug(f"{log_prefix} Raw LLM Response: '{cleaned[:200]}...'")
    parsed_json = None
    try:
        if cleaned.startswith("{") and cleaned.endswith("}"): parsed_json = json.loads(cleaned)
        else:
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
            if json_match: parsed_json = json.loads(json_match.group(1).strip())
            else: logger.error(f"{log_prefix} Response not valid JSON or fenced JSON: {cleaned}"); return None
        if not isinstance(parsed_json, dict): logger.error(f"{log_prefix} Parsed result not dict: {parsed_json}"); return None
        if "classification" in parsed_json:
            cls_value = parsed_json["classification"]
            if isinstance(cls_value, str) and cls_value.upper() in ["CHAT", "ASK", "WORK"]: logger.info(f"{log_prefix} Parsed classification: {cls_value.upper()}"); return parsed_json
            else: logger.error(f"{log_prefix} Invalid classification value: '{cls_value}'"); return None
        elif parsed_json.get("action") == "recall_memory":
            if "query_text" in parsed_json: logger.info(f"{log_prefix} Parsed memory recall params: {parsed_json}"); return parsed_json
            else: logger.error(f"{log_prefix} Missing 'query_text' in recall_memory action."); return None
        else: logger.error(f"{log_prefix} Parsed JSON unexpected format: {parsed_json}"); return None
    except Exception as parse_e: logger.error(f"{log_prefix} Error parsing LLM response: {parse_e}", exc_info=True); return None

def invoke_manager_ack(session_id: str, user_message: str, history_context: List[Dict]) -> str:
    """ Generates a brief user acknowledgement. """
    invocation_role = "Manager ACK"; log_prefix=f"[{session_id}] {invocation_role}"; logger.info(f"{log_prefix}: Preparing...")
    if not VERTEX_AI_AVAILABLE: logger.warning(f"{log_prefix}: Vertex unavailable. Fallback."); return "OK. Processing..."
    if not RIX_BRAIN_LOADED or not all([rix_config, rix_utils]): logger.error(f"{log_prefix}: Rix core missing. Fallback."); return "OK. Processing..."
    try:
        model_name = rix_config.get_config("MANAGER_MODEL"); temp = float(rix_config.get_config("MANAGER_TEMPERATURE", 0.6))
        sdir = _get_souls_dir(); soul_path = sdir / "manager_v3_21.json" if sdir else None # Use new path
        base_prompt = rix_utils.load_soul_prompt(soul_path) if soul_path else "You are Rix."
        if not model_name or not base_prompt: raise ValueError("Manager ACK model/soul missing.")
        hist_snip = history_context[-4:]
        hist_str = "\n".join(rix_utils.format_history_snippet(hist_snip)) if rix_utils else "History Unavailable"
        prompt = (f"{base_prompt}\n\n# Task: Generate Short User Acknowledgement\nUser Message: \"{user_message}\"\nContext:\n{hist_str}\n\n## Output (Strict):\nONLY brief ack message.")
        vertex_contents = _convert_history_to_vertex_content([{'role':'user', 'parts':[{'text':prompt}]}])
        resp_txt, _ = _invoke_llm_with_retry(model_name, vertex_contents, temp, session_id, invocation_role)
        clean_resp = resp_txt.strip() if resp_txt else "";
        if not clean_resp: logger.warning(f"{log_prefix}: Empty ACK. Fallback."); return "OK. Processing..."
        logger.info(f"{log_prefix} Generated ACK: '{clean_resp[:50]}...'")
        return clean_resp
    except Exception as e: logger.error(f"{log_prefix} Error: {e}. Fallback.", exc_info=True); return "OK. Processing... (Error)"

def invoke_manager_dispatch(session_id: str, user_message: str, history_context: List[Dict]) -> Tuple[str, Optional[str]]:
    """ Generates the internal dispatch message for the Thinker. """
    invocation_role = "Manager Dispatch"; log_prefix = f"[{session_id}] {invocation_role}"; logger.info(f"{log_prefix}: Preparing...")
    fallback_instr = f"@Thinker: [Fallback] Analyze request ('{user_message[:50]}...')"; fallback_target = "@Thinker"
    if not VERTEX_AI_AVAILABLE: logger.warning(f"{log_prefix}: Vertex unavailable. Fallback."); return fallback_instr, fallback_target
    if not RIX_BRAIN_LOADED or not all([rix_config, rix_utils]): logger.error(f"{log_prefix}: Rix core missing. Fallback."); return fallback_instr, fallback_target
    try:
        model_name = rix_config.get_config("MANAGER_MODEL"); temp = float(rix_config.get_config("MANAGER_TEMPERATURE", 0.5))
        sdir = _get_souls_dir(); soul_path = sdir / "manager_v3_21.json" if sdir else None # Use new path
        base_prompt = rix_utils.load_soul_prompt(soul_path) if soul_path else "You are Rix."
        if not model_name or not base_prompt: raise ValueError("Manager Dispatch model/soul missing.")
        ctx = _get_context_for_llm(session_id, user_message, history_context, role="Manager")
        hist_str = "\n".join(ctx.get("recent_history_snippet", [])); mem_str = "\n".join(ctx.get("relevant_memories", [])); hints = json.dumps(ctx.get("personal_memory_hints", {}))
        prompt = (f"{base_prompt}\n\n# Task: Internal Dispatch Instruction\nCtx:\n* Hist:\n{hist_str or 'None'}\n* Mem:\n{mem_str or 'None'}\n* User Msg: {user_message}\n* Hints:{hints}\n\n## Output (Strict):\nInstruction for Thinker. MUST start EXACTLY with '@Thinker:'.")
        logger.debug(f"{log_prefix} Dispatch prompt (Len: {len(prompt)}).")
        hist_snip = history_context[-4:]
        vtx_hist = _convert_history_to_vertex_content(hist_snip)
        final_cont = _convert_history_to_vertex_content([{'role': 'user', 'parts': [{'text': prompt}]}]) + vtx_hist
        resp_txt, _ = _invoke_llm_with_retry(model_name, final_cont, temp, session_id, invocation_role, tools=None)
        target = None; resp_strip = resp_txt.strip() if resp_txt else ""; final_dispatch = None; tag_marker = "@Thinker:"
        marker_pos = resp_strip.find(tag_marker)
        if marker_pos != -1:
            potential_instr = resp_strip[marker_pos + len(tag_marker):].strip()
            if potential_instr and len(potential_instr) > 5: final_dispatch = f"{tag_marker} {potential_instr}"; target = "@Thinker"; logger.info(f"{log_prefix} Valid dispatch extracted.")
            else: logger.warning(f"{log_prefix} Found marker but invalid content after: '{potential_instr[:100]}...'")
        else: logger.warning(f"{log_prefix} @Thinker: marker not found: '{resp_strip[:100]}...'")
        if final_dispatch is None: logger.error(f"{log_prefix} Failed validation. Fallback."); final_dispatch = fallback_instr; target = fallback_target
        return final_dispatch, target
    except Exception as e: logger.error(f"{log_prefix} Error: {e}. Fallback.", exc_info=True); return fallback_instr, fallback_target

def invoke_thinker_plan(session_id: str, manager_dispatch_content: str, history_context: List[Dict], tool_schemas: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """ Invokes the Thinker LLM for planning or requesting NATIVE tool calls. """
    invocation_role = "Thinker Plan/FnCall"; log_prefix = f"[{session_id}] {invocation_role}"; logger.info(f"{log_prefix}: Preparing...")
    if not VERTEX_AI_AVAILABLE: raise LLMInvocationError("Vertex AI SDK missing.", role=invocation_role)
    if not RIX_BRAIN_LOADED or not all([rix_config, rix_utils, Rix_tools_play]): raise LLMInvocationError("Rix core components missing.", role=invocation_role)
    try:
        model_name = rix_config.get_config("THINKER_MODEL"); temp = float(rix_config.get_config("THINKER_TEMPERATURE", 0.5))
        sdir = _get_souls_dir(); soul_path = sdir / "thinker_v4_0.json" if sdir else None # Use new path/name
        base_prompt = rix_utils.load_soul_prompt(soul_path) if soul_path else None
        if not model_name or not base_prompt: raise ValueError("Thinker Plan model or soul prompt missing.")
        vertex_tools: Optional[List[VertexTool]] = None
        if tool_schemas:
            try:
                function_declarations: List[FunctionDeclaration] = []
                for schema in tool_schemas:
                    if isinstance(schema, dict) and schema.get("name") and schema.get("description"):
                        params_schema = schema.get("parameters", {"type": "object", "properties": {}, "required": []})
                        if isinstance(params_schema, dict):
                            try: function_declarations.append(FunctionDeclaration(name=schema["name"], description=schema["description"], parameters=params_schema))
                            except Exception as decl_e: logger.error(f"{log_prefix} Fail FnDecl '{schema.get('name')}': {decl_e}", exc_info=True)
                        else: logger.warning(f"{log_prefix} Invalid params schema '{schema.get('name')}'.")
                    else: logger.warning(f"{log_prefix} Skip schema missing name/desc: {schema.get('name')}")
                if function_declarations: vertex_tools = [VertexTool(function_declarations=function_declarations)]; logger.info(f"{log_prefix} Prepared {len(function_declarations)} Vertex Fn Decls.")
                else: logger.warning(f"{log_prefix} No valid Fn Decls created.")
            except Exception as e: logger.error(f"{log_prefix} Fail Vertex Tool prep: {e}", exc_info=True); raise LLMInvocationError(f"Vertex Tool prep failed: {e}", role=invocation_role, original_exception=e)
        dispatch = manager_dispatch_content.replace("@Thinker:", "").strip()
        hist_snip = history_context[-8:]
        ctx = _get_context_for_llm(session_id, dispatch, hist_snip, role="Thinker")
        hist_str = "\n".join(ctx.get("recent_history_snippet", [])); mem_str = "\n".join(ctx.get("relevant_memories", []))
        prompt = base_prompt.replace("{{MANAGER_INSTRUCTION_PLACEHOLDER}}", dispatch).replace("{{RECENT_HISTORY_PLACEHOLDER}}", hist_str or "None").replace("{{FAISS_MEMORIES_PLACEHOLDER}}", mem_str or "None").replace("{{RECENT_LESSONS_PLACEHOLDER}}", mem_str or "None")
        final_cont = _convert_history_to_vertex_content([{'role': 'user', 'parts': [{'text': prompt}]}] + hist_snip)
        text_resp, fn_call_list = _invoke_llm_with_retry(model_name, final_cont, temp, session_id, invocation_role, tools=vertex_tools)
        if fn_call_list:
            logger.info(f"{log_prefix} LLM requested {len(fn_call_list)} function call(s).")
            if text_resp: logger.warning(f"{log_prefix} Text ignored due to function call.")
            return None, fn_call_list
        elif text_resp:
            resp_strip = text_resp.strip()
            if resp_strip.startswith("@Manager:"): logger.info(f"{log_prefix} LLM provided @Manager report."); return resp_strip, None
            else: logger.warning(f"{log_prefix} Text missing '@Manager:'. Prepending."); return f"@Manager: {resp_strip}", None
        else: logger.error(f"{log_prefix} LLM returned EMPTY response. Check P0.4.Fix3."); raise LLMInvocationError("LLM returned empty response.", role=invocation_role)
    except LLMInvocationError as e: logger.error(f"{log_prefix} LLM Error: {e}", exc_info=False); raise e
    except Exception as e: logger.exception(f"{log_prefix} Unexpected error: {e}"); raise LLMInvocationError(f"Invoke unexpected err: {e}", role=invocation_role, original_exception=e)

def invoke_model_with_function_response(session_id: str, history_including_call_and_response: List[Dict]) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """ Invokes LLM after a tool run to synthesize results or decide next step. """
    invocation_role = "FuncResp->Synth"; log_prefix = f"[{session_id}] {invocation_role}"; logger.info(f"{log_prefix}: Preparing response synthesis/next step...")
    if not VERTEX_AI_AVAILABLE: raise LLMInvocationError("Vertex AI SDK missing.", role=invocation_role)
    if not RIX_BRAIN_LOADED or not rix_config: raise LLMInvocationError("Rix config unavailable.", role=invocation_role)
    try:
        model_name = rix_config.get_config("THINKER_MODEL"); temp = float(rix_config.get_config("THINKER_TEMPERATURE", 0.5))
        if not model_name: raise ValueError("Synthesis model (Thinker) missing.")
        vtx_hist = _convert_history_to_vertex_content(history_including_call_and_response)
        if not vtx_hist: raise ValueError("History conversion failed.")
        if vtx_hist[-1].role != "function": logger.warning(f"{log_prefix}: History doesn't end with 'function'. Last: {vtx_hist[-1].role}.")
        tool_schemas = Rix_tools_play.get_tool_schemas() if Rix_tools_play else []
        vertex_tools: Optional[List[VertexTool]] = None
        if tool_schemas:
             try:
                 function_declarations = [FunctionDeclaration(name=s["name"], description=s["description"], parameters=s.get("parameters", {})) for s in tool_schemas if s.get("name") and s.get("description")]
                 if function_declarations: vertex_tools = [VertexTool(function_declarations=function_declarations)]; logger.info(f"{log_prefix} Providing {len(function_declarations)} tools for next step.")
             except Exception as e: logger.error(f"{log_prefix} Failed Vertex Tool prep synth: {e}")
        logger.info(f"{log_prefix} Calling model post-tool run...")
        final_txt, further_calls_list = _invoke_llm_with_retry(model_name, vtx_hist, temp, session_id, invocation_role, tools=vertex_tools)
        if further_calls_list: logger.info(f"{log_prefix} LLM requested FURTHER tool calls ({len(further_calls_list)})."); return None, further_calls_list
        elif final_txt:
            logger.info(f"{log_prefix} LLM generated synthesized text (Length: {len(final_txt)}).")
            resp_strip = final_txt.strip()
            if not resp_strip.startswith("@Manager:"): logger.warning(f"{log_prefix} Synth text missing '@Manager:'. Prepending."); resp_strip = f"@Manager: {resp_strip}"
            return resp_strip, None
        else: logger.error(f"{log_prefix} LLM returned NO text/calls after tool response."); raise LLMInvocationError("LLM failed post-tool synth.", role=invocation_role)
    except LLMInvocationError as e: logger.error(f"{log_prefix} LLM Error: {e}"); raise e
    except Exception as e: logger.exception(f"{log_prefix} Unexpected error: {e}"); raise LLMInvocationError(f"Unexpected synth err: {e}", role=invocation_role, original_exception=e)

def invoke_manager_finalize(session_id: str, thinker_report_or_chat_input: str, history_context: List[Dict]) -> Tuple[str, Optional[str]]:
    """ Invokes Manager LLM for final user response or CHAT. """
    invocation_role = "Manager Finalize"; log_prefix = f"[{session_id}] {invocation_role}"; logger.info(f"{log_prefix}: Preparing final response...")
    if not VERTEX_AI_AVAILABLE: raise LLMInvocationError("Vertex AI SDK missing.", role=invocation_role)
    if not RIX_BRAIN_LOADED or not all([rix_config, rix_utils]): raise LLMInvocationError("Rix core components missing.", role=invocation_role)
    try:
        model_name = rix_config.get_config("MANAGER_MODEL"); temp = float(rix_config.get_config("MANAGER_TEMPERATURE", 0.7)) # Slightly higher temp?
        sdir = _get_souls_dir(); soul_path = sdir / "manager_v3_21.json" if sdir else None # Use new path
        base_prompt = rix_utils.load_soul_prompt(soul_path) if soul_path else None
        if not model_name or not base_prompt: raise ValueError("Manager Finalize model/soul missing.")
        input_content = thinker_report_or_chat_input.replace("@Manager:", "").strip()
        ctx = _get_context_for_llm(session_id, input_content[:500], history_context, role="Manager")
        hist_snip = history_context[-8:]
        hist_str = "\n".join(ctx.get("recent_history_snippet", [])); mem_str = "\n".join(ctx.get("relevant_memories", [])); hints = json.dumps(ctx.get("personal_memory_hints", {}))
        prompt = base_prompt.replace("{{THINKER_REPORT_PLACEHOLDER}}", input_content).replace("{{THINKER_INPUT_PLACEHOLDER}}", input_content).replace("{{RECENT_HISTORY_PLACEHOLDER}}", hist_str or "None").replace("{{FAISS_MEMORIES_PLACEHOLDER}}", mem_str or "None").replace("{{PERSONAL_HINTS_PLACEHOLDER}}", hints)
        if "Output ONLY the complete, final message" not in prompt: prompt += "\n\n**Output (Strict):**\nOutput **ONLY** the complete, final message to user."
        final_cont = _convert_history_to_vertex_content([{'role': 'user', 'parts': [{'text': prompt}]}] + hist_snip)
        resp_txt, _ = _invoke_llm_with_retry(model_name, final_cont, temp, session_id, invocation_role, tools=None)
        target = "@User"; final_resp = resp_txt.strip() if resp_txt else ""
        if final_resp.startswith("@User:"): logger.warning(f"{log_prefix} Removing leading '@User:' tag."); final_resp = final_resp[len("@User:"):].strip()
        if not final_resp: logger.error(f"{log_prefix} LLM returned EMPTY! Fallback."); final_resp = "Processing complete."
        logger.info(f"{log_prefix} Generated final response (Length: {len(final_resp)}).")
        return final_resp, target
    except LLMInvocationError as e: logger.error(f"{log_prefix} LLM Error: {e}"); raise e
    except Exception as e: logger.exception(f"{log_prefix} Unexpected error: {e}"); raise LLMInvocationError(f"Unexpected finalize err: {e}", role=invocation_role, original_exception=e)

def invoke_memory_writer(session_id: str, interaction_snippet: List[Dict], turn_status: str, flow_type: Literal["CHAT", "WORK", "ASK", "ERROR", "REFLECTION"] = "CHAT") -> str:
    """ Invokes LLM for summary/reflection, saves to Cloud SQL. """
    invocation_role = "MemWriter"; log_prefix = f"[{session_id}] {invocation_role}({flow_type})"; logger.info(f"{log_prefix}: Preparing memory/reflection...")
    fallback_summary = f"[Placeholder Memory - Status: {turn_status}, Flow: {flow_type}]"
    if not VERTEX_AI_AVAILABLE: logger.error(f"{log_prefix}: Vertex unavailable."); return fallback_summary + " (Vertex Unavailable)"
    if not RIX_BRAIN_LOADED or not all([rix_config, rix_utils, rix_vector_pg]): logger.error(f"{log_prefix}: Rix core missing."); return fallback_summary + " (Rix Core Unavailable)"
    try:
        model_name = rix_config.get_config("MEMORY_WRITER_MODEL"); temp = float(rix_config.get_config("MEMORY_WRITER_TEMPERATURE", 0.4))
        sdir = _get_souls_dir(); soul_path = sdir / "memory_writer_v1_0.json" if sdir else None # Use new path
        soul_prompt_template = rix_utils.load_soul_prompt(soul_path) if soul_path else None
        if not model_name or not soul_prompt_template: raise ValueError("Memory Writer model/soul missing.")
        snip_hist = interaction_snippet[-10:]
        interact_str = "\n".join(rix_utils.format_history_snippet(snip_hist)) if rix_utils else "History Unavailable."
        sysp = soul_prompt_template.replace("{{INTERACTION_SNIPPET_PLACEHOLDER}}", interact_str or "None").replace("{{TURN_STATUS_PLACEHOLDER}}", turn_status)
        if "{{FLOW_TYPE_PLACEHOLDER}}" in sysp: sysp = sysp.replace("{{FLOW_TYPE_PLACEHOLDER}}", flow_type)
        else: logger.warning(f"{log_prefix} Mem writer soul missing FLOW_TYPE placeholder.")
        vertex_contents = _convert_history_to_vertex_content([{'role':'user', 'parts':[{'text':sysp}]}])
        resp_txt, _ = _invoke_llm_with_retry(model_name, vertex_contents, temp, session_id, invocation_role, tools=None, max_retries=1)
        summary = resp_txt.strip() if resp_txt else ""
        logger.info(f"{log_prefix} Memory text generated (Length: {len(summary)}).")
        if not summary: logger.warning(f"{log_prefix} Empty summary! Fallback."); summary = fallback_summary + " (LLM Empty)"
        memory_category = "reflection" if flow_type == "REFLECTION" else f"{flow_type.lower()}_summary"
        logger.info(f"{log_prefix} Saving memory (Category: {memory_category})...")
        meta = {"source": f"auto_mem_{flow_type.lower()}", "session_id": session_id, "timestamp_utc": (rix_utils.get_current_timestamp() if rix_utils else datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"), "turn_status": turn_status, "source_indices": [m.get("index") for m in snip_hist if m.get("index") is not None], "text_length": len(summary)}
        ok, mem_id = rix_vector_pg.add_to_vector_memory(memory_text=summary, metadata=meta, memory_category=memory_category)
        if ok: logger.info(f"{log_prefix} Memory saved (ID: {mem_id}).")
        else: logger.error(f"{log_prefix} Failed save memory.")
        return summary
    except LLMInvocationError as e: logger.error(f"{log_prefix} LLM Error: {e}"); return f"[Error generating memory: {e}]"
    except Exception as e: logger.exception(f"{log_prefix} Unexpected Error: {e}"); return f"[Error generating memory: {e}]"

# --- Deprecated Functions ---
def invoke_classifier(*args, **kwargs) -> str:
     logger.warning("DEPRECATED: invoke_classifier called. Use invoke_classifier_agent.")
     return "WORK"
def invoke_thinker_report(*args, **kwargs) -> Tuple[str, Optional[str]]:
     logger.warning("DEPRECATED: invoke_thinker_report called. Use invoke_model_with_function_response or re-plan.")
     return "@Manager: [Error - Deprecated reporting function called]", "@Manager"

print(f"--- Module Defined: {__name__} (V1.9.1 - V52 Imports) ---", flush=True)
# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\services\vertex_llm.py ---