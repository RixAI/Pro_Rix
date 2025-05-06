# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\services\firestore_history.py ---
# Version: 2.3.0 (V52.0 - Updated Imports) - Formerly rix_history_manager.py
# Author: Vishal Sharma / Gemini
# Date: May 6, 2025
# Description: Manages Rix conversation history using Google Cloud Firestore via RixState.

print(f"--- Loading Module: {__name__} (V2.3.0 - V52 Imports) ---", flush=True)

import logging
import datetime
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Literal

# --- Rix Core Module Imports (V52 Structure) ---
try:
    from Rix_Brain.core import global_state as RixState # Alias
    from Rix_Brain.core import config_manager as rix_config # Alias
    from Rix_Brain.core import utils as rix_utils # Alias
    RIX_BRAIN_LOADED = True # Assume if these import, core is available
    print(f"--- {__name__}: Rix Brain core imports successful.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Import Fail: {e}", flush=True)
    RIX_BRAIN_LOADED = False
    # Define fallback if core failed, though functions below will likely error
    RixState, rix_config, rix_utils = None, None, None

# --- Google Cloud Firestore Client ---
FIRESTORE_CLIENT_LOADED = False
firestore = None
gcp_exceptions = None
SERVER_TIMESTAMP = object() # Default dummy object

try:
    from google.cloud import firestore
    from google.api_core import exceptions as gcp_exceptions
    SERVER_TIMESTAMP = firestore.SERVER_TIMESTAMP # Assign real sentinel
    FIRESTORE_CLIENT_LOADED = True
    logger = logging.getLogger("Rix_Heart")
    logger.info("Firestore History Service: Firestore client library imported.")
    print(f"--- {__name__}: Firestore client library imported.", flush=True)
except ImportError as e:
    logger = logging.getLogger("Rix_Heart")
    logger.critical(f"FATAL: Firestore library not found: {e}. History will FAIL.", exc_info=True)
    print(f"--- {__name__}: Firestore client library IMPORT FAILED.", flush=True)
    # Keep dummy SERVER_TIMESTAMP

# --- Logging Setup ---
logger = logging.getLogger("Rix_Heart") # Use central logger name

# --- Constants and Locks ---
try:
    # Use config manager to get collection name, provide default
    HISTORY_BASE_COLLECTION = rix_config.get_config("FIRESTORE_HISTORY_COLLECTION", "rix_chat_sessions")
except Exception:
     HISTORY_BASE_COLLECTION = "rix_chat_sessions" # Fallback
     logger.error("Could not get FIRESTORE_HISTORY_COLLECTION from config, using default.")

SESSION_LOCKS: Dict[str, threading.Lock] = {}
LOCK_FOR_SESSION_LOCKS = threading.Lock()

def _get_session_lock(session_id: str) -> threading.Lock:
    """Gets or creates a lock for a specific session ID."""
    with LOCK_FOR_SESSION_LOCKS:
        if session_id not in SESSION_LOCKS:
            SESSION_LOCKS[session_id] = threading.Lock()
        return SESSION_LOCKS[session_id]

# --- Helper to get Firestore client from RixState ---
def _get_db_client() -> Optional[firestore.Client]: # Return Optional
    """Retrieves the Firestore client from RixState."""
    if not RIX_BRAIN_LOADED or not RixState:
         logger.error("RixState not available for DB client lookup.")
         return None
    if not FIRESTORE_CLIENT_LOADED:
         logger.error("Firestore library not loaded.")
         return None

    db_client = getattr(RixState, 'firestore_db_client', getattr(RixState, 'firestore_db', None)) # Check common names
    if not db_client:
        logger.error("Firestore client not found in RixState.")
        return None
    # Relax type check slightly, rely on operations failing if it's wrong type
    # if not isinstance(db_client, firestore.Client):
    #     logger.error(f"Firestore client in RixState is not expected type: {type(db_client).__name__}")
    #     return None
    return db_client

# --- Helper to get session/message collection refs ---
def _get_session_messages_collection_ref(session_id: str) -> Optional[firestore.CollectionReference]:
    """Helper to get the Firestore collection reference for a session's messages."""
    db = _get_db_client()
    if not db: return None # Error logged in _get_db_client
    if not RIX_BRAIN_LOADED or not rix_utils:
        logger.error("Rix Utils not available for session ID validation.")
        return None
    if not rix_utils.validate_session_id(session_id, "_get_session_messages_collection_ref"):
         logger.error(f"Invalid session_id format: {session_id}")
         return None
    return db.collection(HISTORY_BASE_COLLECTION).document(session_id).collection("messages")

def _get_session_doc_ref(session_id: str) -> Optional[firestore.DocumentReference]:
    """Helper to get the Firestore document reference for a session."""
    db = _get_db_client()
    if not db: return None
    if not RIX_BRAIN_LOADED or not rix_utils:
        logger.error("Rix Utils not available for session ID validation.")
        return None
    if not rix_utils.validate_session_id(session_id, "_get_session_doc_ref"):
        logger.error(f"Invalid session_id format: {session_id}")
        return None
    return db.collection(HISTORY_BASE_COLLECTION).document(session_id)

# --- Core History Functions ---

def load_history(session_id: str) -> List[Dict[str, Any]]:
    """Loads the entire conversation history for a session from Firestore."""
    log_prefix = f"[{session_id}] load_history"
    logger.info(f"{log_prefix} Loading history...")
    history: List[Dict[str, Any]] = []
    if not RIX_BRAIN_LOADED or not rix_utils:
        logger.error(f"{log_prefix} Rix Utils unavailable."); return []

    if not rix_utils.validate_session_id(session_id, "load_history"):
        logger.error(f"{log_prefix} Invalid session ID format."); return []

    try:
        messages_ref = _get_session_messages_collection_ref(session_id)
        if not messages_ref: raise ConnectionError("Failed to get messages collection reference.")

        # Use Query object for ordering
        query = messages_ref.order_by("index", direction=firestore.Query.ASCENDING)
        docs_stream = query.stream() # Get stream from query

        for doc in docs_stream:
            entry = doc.to_dict()
            if entry:
                # --- Index Handling ---
                if 'index' not in entry:
                    try: entry['index'] = int(doc.id)
                    except ValueError: logger.warning(f"{log_prefix} Doc ID '{doc.id}' not int index, skipping."); continue
                else:
                    try: entry['index'] = int(entry['index'])
                    except (ValueError, TypeError): logger.warning(f"{log_prefix} Stored index '{entry['index']}' not int, using doc ID '{doc.id}'."); entry['index'] = int(doc.id) # Fallback

                # --- Timestamp Handling ---
                ts = entry.get('timestamp')
                if isinstance(ts, datetime.datetime):
                    # Format Firestore timestamp to ISO string Z format
                    entry['timestamp'] = ts.isoformat(timespec='microseconds') + 'Z'
                elif hasattr(ts, 'isoformat'): # Handle other potential timestamp objects
                    try: entry['timestamp'] = ts.isoformat(timespec='microseconds') + 'Z'
                    except Exception: entry['timestamp'] = str(ts) # Fallback to string
                elif ts is not None: entry['timestamp'] = str(ts) # Convert unknown non-None to string

                # --- Content Handling (Ensure basic types or string) ---
                content = entry.get('content')
                if not isinstance(content, (str, int, float, bool, list, dict, type(None))):
                    logger.warning(f"{log_prefix} Entry {entry['index']}: Non-standard content type '{type(content).__name__}' found during load, converting to string.")
                    entry['content'] = str(content)

                history.append(entry)

        logger.info(f"{log_prefix} Loaded {len(history)} entries.")
        return history

    except gcp_exceptions.NotFound:
        logger.info(f"{log_prefix} No history found for session."); return []
    except ConnectionError as conn_e:
        logger.error(f"{log_prefix} Firestore client/connection issue: {conn_e}"); return []
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error loading history: {e}", exc_info=True); return []


def append_message(session_id: str, message_dict: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
    """
    Appends a message to the session's history in Firestore atomically.
    Returns (success_status, assigned_index).
    """
    log_prefix = f"[{session_id}] append_message"
    if not RIX_BRAIN_LOADED or not rix_utils: logger.error(f"{log_prefix} Rix Utils unavailable."); return False, None
    if not rix_utils.validate_session_id(session_id, "append_message"): logger.error(f"{log_prefix} Invalid session ID format."); return False, None

    session_lock = _get_session_lock(session_id)
    with session_lock:
        logger.debug(f"{log_prefix} [APPEND_FS] Lock acquired.")
        db = None
        try:
            db = _get_db_client()
            if not db: raise ConnectionError("Firestore client unavailable for append.")

            messages_ref = _get_session_messages_collection_ref(session_id)
            if not messages_ref: raise ConnectionError("Failed to get messages collection reference for append.")

            transaction = db.transaction()

            @firestore.transactional # type: ignore # Decorator exists if library loaded
            def _append_in_transaction(transaction, messages_ref_tx, new_message_data_payload):
                """Transaction logic to find next index and set the document."""
                # Query for the document with the highest index
                query = messages_ref_tx.order_by("index", direction=firestore.Query.DESCENDING).limit(1)
                docs = list(query.stream(transaction=transaction)) # Execute query within transaction
                next_index = 0
                if docs:
                    try:
                        last_index = docs[0].to_dict().get("index")
                        # Attempt conversion to int, default to 0 on failure or non-numeric
                        next_index = int(last_index) + 1 if isinstance(last_index, (int, float)) else 0
                        next_index = max(0, next_index) # Ensure non-negative
                    except (ValueError, TypeError, KeyError, AttributeError) as index_e:
                        logger.warning(f"{log_prefix} Error reading last index ('{last_index}'): {index_e}. Resetting next index to 0.")
                        next_index = 0
                else:
                    # No documents found, start index at 0
                    next_index = 0

                logger.debug(f"{log_prefix} Determined next index: {next_index}")

                # Prepare data payload
                new_message_data_payload["index"] = next_index
                if FIRESTORE_CLIENT_LOADED and SERVER_TIMESTAMP is not object:
                    new_message_data_payload["timestamp"] = SERVER_TIMESTAMP
                else:
                    logger.error(f"{log_prefix} Firestore SERVER_TIMESTAMP unavailable. Using client time.")
                    new_message_data_payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc)

                # Ensure Firestore compatibility
                serializable_data = {}
                for k, v in new_message_data_payload.items():
                    is_standard_type = isinstance(v, (str, int, float, bool, list, dict, type(None), datetime.datetime))
                    is_sentinel = FIRESTORE_CLIENT_LOADED and v == SERVER_TIMESTAMP
                    if is_standard_type or is_sentinel:
                        serializable_data[k] = v
                    else:
                        logger.warning(f"{log_prefix} Field '{k}' type {type(v).__name__} not standard/sentinel. Converting to string.")
                        serializable_data[k] = str(v)

                # Use the index as the document ID (string format)
                doc_ref = messages_ref_tx.document(str(next_index))
                transaction.set(doc_ref, serializable_data)
                logger.debug(f"{log_prefix} Transaction set: Doc ID '{next_index}', Index Field: {next_index}")
                return next_index # Return the assigned index

            # Execute the transaction
            assigned_index = _append_in_transaction(transaction, messages_ref, message_dict.copy())

            if assigned_index is not None:
                logger.info(f"{log_prefix} Appended index {assigned_index} (Actor:{message_dict.get('actor')}, Type:{message_dict.get('message_type')}, Target:{message_dict.get('target_recipient')}).")
                return True, assigned_index
            else:
                # This case might occur if the transactional function itself returns None implicitly,
                # although usually an exception would be raised on failure.
                logger.error(f"{log_prefix} Transaction completed but returned None index.")
                return False, None

        except ConnectionError as conn_e:
            logger.error(f"{log_prefix} Firestore client/connection issue during append: {conn_e}")
            return False, None
        except Exception as e:
            # This catches errors during the transaction commit or other unexpected issues
            logger.error(f"{log_prefix} Unexpected error appending message: {e}", exc_info=True)
            return False, None
        finally:
            logger.debug(f"{log_prefix} [APPEND_FS] Lock released.")


def edit_message(session_id: str, message_index: int, update_data: Dict[str, Any]) -> bool:
    """Edits a specific message in the history using Firestore update."""
    log_prefix = f"[{session_id}] edit_message"
    logger.info(f"{log_prefix} Editing index {message_index}...")
    if not RIX_BRAIN_LOADED or not rix_utils: logger.error(f"{log_prefix} Rix Utils unavailable."); return False
    if not rix_utils.validate_session_id(session_id, "edit_message"): logger.error(f"{log_prefix} Invalid session ID."); return False
    if not isinstance(message_index, int) or message_index < 0: logger.error(f"{log_prefix} Invalid index {message_index}."); return False
    if not update_data or not isinstance(update_data, dict): logger.error(f"{log_prefix} No update data."); return False

    session_lock = _get_session_lock(session_id)
    with session_lock:
        logger.debug(f"{log_prefix} [EDIT_FS] Lock acquired.")
        try:
            messages_ref = _get_session_messages_collection_ref(session_id)
            if not messages_ref: raise ConnectionError("Failed to get messages collection reference.")

            doc_id = str(message_index) # Use index as document ID
            doc_ref = messages_ref.document(doc_id)

            # Check if document exists before trying to update
            # doc_snapshot = doc_ref.get()
            # if not doc_snapshot.exists:
            #     logger.error(f"{log_prefix} Document index {message_index} not found for editing.")
            #     return False
            # Edit: Use update() directly which fails if doc doesn't exist (simpler check)

            update_payload = {}
            EXCLUDED_EDIT_KEYS = ["index", "timestamp", "actor", "name", "message_type", "target_recipient", "memory_id"]
            for key, value in update_data.items():
                if key in EXCLUDED_EDIT_KEYS:
                    logger.warning(f"{log_prefix} Skipping edit of excluded field '{key}'.")
                    continue
                # Ensure value is serializable
                is_basic_type = isinstance(value, (str, int, float, bool, list, dict, type(None), datetime.datetime))
                if is_basic_type:
                    serializable_value = value
                else:
                    logger.warning(f"{log_prefix} Edit field '{key}' not basic type ({type(value).__name__}). Converting to string.")
                    serializable_value = str(value)
                update_payload[key] = serializable_value

            if not update_payload: logger.warning(f"{log_prefix} No valid fields for update."); return False

            # Add last edited timestamp
            if FIRESTORE_CLIENT_LOADED and SERVER_TIMESTAMP is not object:
                update_payload['last_edited_timestamp'] = SERVER_TIMESTAMP
            else:
                logger.error(f"{log_prefix} SERVER_TIMESTAMP unavailable. Using client time for edit.")
                update_payload['last_edited_timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z'

            # Perform the update
            doc_ref.update(update_payload)
            logger.info(f"{log_prefix} Successfully edited message index {message_index}.")
            return True

        except gcp_exceptions.NotFound:
            logger.error(f"{log_prefix} Document index {message_index} not found during update attempt.")
            return False
        except ConnectionError as conn_e:
            logger.error(f"{log_prefix} Firestore client/connection issue during edit: {conn_e}")
            return False
        except Exception as e:
            logger.error(f"{log_prefix} Unexpected error editing message index {message_index}: {e}", exc_info=True)
            return False
        finally:
            logger.debug(f"{log_prefix} [EDIT_FS] Lock released.")


def delete_message(session_id: str, message_index: int) -> bool:
    """Deletes a specific message document by its index."""
    log_prefix = f"[{session_id}] delete_message"
    logger.warning(f"{log_prefix} Deleting index {message_index}...")
    if not RIX_BRAIN_LOADED or not rix_utils: logger.error(f"{log_prefix} Rix Utils unavailable."); return False
    if not rix_utils.validate_session_id(session_id, "delete_message"): logger.error(f"{log_prefix} Invalid session ID."); return False
    if not isinstance(message_index, int) or message_index < 0: logger.error(f"{log_prefix} Invalid index {message_index}."); return False

    session_lock = _get_session_lock(session_id)
    with session_lock:
        logger.debug(f"{log_prefix} [DELETE_MSG_FS] Lock acquired.")
        try:
            messages_ref = _get_session_messages_collection_ref(session_id)
            if not messages_ref: raise ConnectionError("Failed to get messages collection reference.")

            doc_id = str(message_index) # Use index as document ID
            doc_ref = messages_ref.document(doc_id)
            doc_ref.delete() # delete() doesn't error if doc doesn't exist
            logger.info(f"{log_prefix} Delete requested/completed for index {message_index}.")
            return True # Assume success even if doc didn't exist
        except ConnectionError as conn_e:
            logger.error(f"{log_prefix} Firestore client/connection issue during delete: {conn_e}")
            return False
        except Exception as e:
            logger.error(f"{log_prefix} Unexpected error deleting message index {message_index}: {e}", exc_info=True)
            return False
        finally:
            logger.debug(f"{log_prefix} [DELETE_MSG_FS] Lock released.")

def delete_session_history(session_id: str) -> bool:
    """Deletes all messages within a session and the session document itself."""
    log_prefix = f"[{session_id}] delete_session_history"
    logger.warning(f"{log_prefix} Attempting DELETE ALL history...")
    if not RIX_BRAIN_LOADED or not rix_utils: logger.error(f"{log_prefix} Rix Utils unavailable."); return False
    if not rix_utils.validate_session_id(session_id, "delete_session_history"): logger.error(f"{log_prefix} Invalid session ID."); return False

    session_lock = _get_session_lock(session_id)
    with session_lock:
        logger.debug(f"{log_prefix} [DELETE_SESS_FS] Lock acquired.")
        db = None
        try:
            db = _get_db_client()
            if not db: raise ConnectionError("Firestore client unavailable for delete session.")

            messages_ref = _get_session_messages_collection_ref(session_id)
            if messages_ref:
                logger.info(f"{log_prefix} Deleting 'messages' subcollection...")
                _delete_collection(db, messages_ref, 100) # Increased batch size
                logger.info(f"{log_prefix} 'messages' subcollection deletion complete.")
            else:
                 logger.warning(f"{log_prefix} Could not get 'messages' collection reference to delete.")


            session_doc_ref = _get_session_doc_ref(session_id)
            if session_doc_ref:
                 logger.info(f"{log_prefix} Deleting session document itself...")
                 session_doc_ref.delete()
                 logger.info(f"{log_prefix} Session document deletion complete.")
            else:
                 logger.warning(f"{log_prefix} Could not get session document reference to delete.")

            logger.info(f"{log_prefix} Successfully deleted session history.")
            return True

        except ConnectionError as conn_e:
            logger.error(f"{log_prefix} Firestore client/connection issue during delete session: {conn_e}")
            return False
        except Exception as e:
            logger.error(f"{log_prefix} Error deleting session history: {e}", exc_info=True)
            return False
        finally:
            logger.debug(f"{log_prefix} [DELETE_SESS_FS] Lock released.")
            # Remove lock from dictionary after session is deleted
            with LOCK_FOR_SESSION_LOCKS:
                SESSION_LOCKS.pop(session_id, None)


# --- Helper to delete collection in batches ---
def _delete_collection(db_client: firestore.Client, coll_ref: firestore.CollectionReference, batch_size: int):
    """Helper to delete all documents in a collection in batches."""
    deleted_total = 0
    while True:
        # Get a batch of documents
        docs = coll_ref.limit(batch_size).stream()
        deleted_count_batch = 0
        batch = db_client.batch()
        for doc in docs:
            logger.debug(f"Scheduling deletion of doc {doc.id} from collection {coll_ref.path}...")
            batch.delete(doc.reference)
            deleted_count_batch += 1

        if deleted_count_batch == 0:
            # No more documents found, deletion complete
            break

        # Commit the batch
        try:
             batch.commit()
             deleted_total += deleted_count_batch
             logger.debug(f"Committed deletion batch of {deleted_count_batch} documents (Total: {deleted_total}).")
        except Exception as e:
             logger.error(f"Error committing deletion batch for {coll_ref.path}: {e}", exc_info=True)
             raise # Re-raise to signal failure to the caller


# --- Utility Function (V52 - Standardized Message Types) ---
def create_history_entry_dict(
    actor: str,
    content: Any,
    message_type: Literal[
        "user_input",             # User typed message
        "system_response",        # Final response/ack/question FOR the user
        "classification_result",  # Output from Classifier (intent)
        "dispatch_instruction",   # Output from Manager (instruction for Thinker)
        "internal_report",        # Text output from Thinker/Tool for Manager/Finalizer
        "internal_question",      # Agent asking another internal agent a question
        "internal_answer",        # Agent answering another internal agent's question
        "tool_action_request",    # Thinker requesting a tool call (content = JSON list of calls)
        "tool_action_result",     # ToolExecutor result (content = JSON result dict)
        "memory_recall_guidance", # Classifier suggesting params for memory recall
        "memory_recall_result",   # MemorySystem returning results
        "reflection_insight",     # Output from MemoryWriter (reflection)
        "system_error"            # Internal error message
    ],
    name: Optional[str] = None, # Optional specific name (e.g., tool name)
    target_recipient: Optional[str] = None, # Use '@AgentName' format or '@User'
    # Tool specific fields (populated when message_type is relevant)
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None, # For tool_action_request
    tool_status: Optional[Literal["success", "failed", "timeout"]] = None, # For tool_action_result
    # Structured data fields (alternative to packing into content)
    function_call_data: Optional[List[Dict]] = None, # For tool_action_request (native call format)
    function_response_data: Optional[Dict] = None, # For tool_action_result (native call format)
    # Error details
    error_details: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Creates a standard history entry dictionary (V52 Structure).
    Timestamp and index are added by append_message.
    """
    entry: Dict[str, Any] = {
        "actor": actor,
        "name": name if name is not None else actor,
        "message_type": message_type,
        "target_recipient": target_recipient,
         # --- Content Handling ---
         # Prioritize structured data if provided for specific types
         # Store primary text/object in 'content' otherwise
    }

    # Add specific structured data based on type
    if message_type == "tool_action_request" and function_call_data:
        entry["function_call_data"] = function_call_data # Store native call structure
        entry["content"] = f"[Requesting {len(function_call_data)} tool call(s)]" # Placeholder content
        if len(function_call_data) == 1:
             entry["tool_name"] = function_call_data[0].get("name")
             entry["tool_args"] = function_call_data[0].get("args")
    elif message_type == "tool_action_result" and function_response_data:
         entry["function_response_data"] = function_response_data # Store native response structure
         entry["tool_name"] = function_response_data.get("name")
         response_detail = function_response_data.get("response", {})
         entry["tool_status"] = response_detail.get("status")
         entry["content"] = json.dumps(response_detail) # Store the inner response dict as JSON string in content
         if response_detail.get("error"): entry["error_details"] = response_detail.get("error")
    else:
        # For other types, ensure content is serializable
        if not isinstance(content, (str, int, float, bool, list, dict, type(None))):
            log.warning(f"Converting non-standard content type '{type(content).__name__}' to string for history entry.")
            entry["content"] = str(content)
        else:
            entry["content"] = content # Assign standard types directly

    # Add optional fields if provided directly (and not already set by structured data)
    if tool_name is not None and "tool_name" not in entry: entry["tool_name"] = tool_name
    if tool_args is not None and "tool_args" not in entry and message_type == "tool_action_request": entry["tool_args"] = tool_args
    if tool_status is not None and "tool_status" not in entry and message_type == "tool_action_result": entry["tool_status"] = tool_status
    if error_details is not None and "error_details" not in entry:
         entry["error_details"] = str(error_details) if isinstance(error_details, Exception) else error_details

    # Remove keys with None values for cleaner storage, EXCEPT target_recipient
    final_entry = {k: v for k, v in entry.items() if v is not None or k == 'target_recipient'}

    return final_entry


print(f"--- Module Defined: {__name__} (V2.3.0 - V52 Imports) ---", flush=True)

# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\services\firestore_history.py ---