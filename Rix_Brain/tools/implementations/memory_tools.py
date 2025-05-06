# --- START OF FULL FILE Rix_Brain/tool_implementations/memory_tools.py ---
# C:\Rix_Dev\Project_Rix\Rix_Brain\tool_implementations\memory_tools.py
# Version: 1.0.3 (V43.1 - Remove additionalProperties from Schema) # Version updated
# Author: Vishal Sharma / Grok 3
# Date: May 2, 2025
# Description: Contains tools for interacting with Rix's long-term memory system (FAISS).
#              Uses ABSOLUTE imports. Fixes Schema validation error.

import logging
from typing import Dict, Any, Optional

print(f"--- LOADING {__name__} (V1.0.3) ---", flush=True) # Use __name__ for clarity

# Attempt to import the memory module using ABSOLUTE path
try: import Rix_Brain.Rix_long_memory_Faiss as Rix_long_memory_Faiss; print("INFO: memory_tools imported Rix_Brain.Rix_long_memory_Faiss successfully.")
except ImportError: print("WARNING: Failed import Rix_Brain.Rix_long_memory_Faiss. Tool will fail."); Rix_long_memory_Faiss = None

logger = logging.getLogger("Rix_Heart") # Use central logger

# --- Tool Schemas (FIXED) ---
TOOL_SCHEMAS = {
     "create_memory_entry": {
        "name": "create_memory_entry",
        "description": "Creates a new text entry in Rix's long-term vector memory (FAISS) for later recall.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The main text content of the memory entry to be stored and embedded."
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional: A JSON object containing key-value pairs for metadata (e.g., {'source': 'user_note', 'importance': 5, 'topic': 'project_alpha'}). Standard keys like 'timestamp_utc' and 'memory_id' are added automatically."
                    # "additionalProperties": True # <<<--- REMOVED THIS INVALID LINE ---<<<
                }
            },
            "required": ["text"]
        }
    }
    # Add other schemas here
}


# --- Tool Function ---
# (Implementation of create_memory_entry remains unchanged from V1.0.2)
def create_memory_entry( text: str, metadata: Optional[Dict[str, Any]] = None, memory_manager: Optional[Any] = None ) -> Dict[str, Any]:
    command="create_memory_entry"; logger.info(f"Tool '{command}' called. Len:{len(text)}. Meta:{metadata}"); meta={"in_len":len(text), "in_meta":metadata}
    if not memory_manager or memory_manager != Rix_long_memory_Faiss or not hasattr(memory_manager,'add_to_vector_memory'): err="Mem Mgr unavail."; logger.error(f"'{command}': {err}"); return { "cmd":command,"status":"failed","error":{"code":"MEM_UNAVAIL","message":err},"message":"Fail: Mem sys unavail.","metadata":meta }
    if not text or not isinstance(text,str): err="Input text required."; logger.error(f"'{command}': {err}"); return { "cmd":command,"status":"failed","error":{"code":"INV_INPUT","message":err},"message":"Fail: Input text required.","metadata":meta }
    final_meta=metadata if isinstance(metadata,dict) else {}; final_meta.setdefault('source','llm_tool_create')
    try:
        success, mem_id = memory_manager.add_to_vector_memory(index=None, text=text, metadata=final_meta)
        if success and mem_id: msg=f"Memory created OK (ID:{mem_id})."; logger.info(f"'{command}': {msg}"); meta["memory_id"]=mem_id; return { "cmd":command,"status":"success","result":{"memory_id":mem_id,"message":msg},"error":None,"message":f"Memory created (ID:{mem_id}).","metadata":meta }
        else: err="Mem sys fail add."; logger.error(f"'{command}': {err}"); return { "cmd":command,"status":"failed","result":{"memory_id":None,"message":"Fail add mem."}, "error":{"code":"MEM_ADD_FAIL","message":err},"message":"Fail: Could not add mem.","metadata":meta }
    except Exception as e: err=f"Unexpected mem creation err:{e}"; logger.exception(f"'{command}': {err}"); return { "cmd":command,"status":"failed","result":None, "error":{"code":"UNEXPECTED_ERR","message":err},"message":f"Fail: Unexpected - {e}","metadata":meta }

# --- END OF FULL FILE Rix_Brain/tool_implementations/memory_tools.py ---