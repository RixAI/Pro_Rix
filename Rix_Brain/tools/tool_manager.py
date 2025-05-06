# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\tools\tool_manager.py ---
# Version: 2.3.0 (V52.0 - Updated Imports) - Formerly Rix_tools_play.py
# Author: Vishal Sharma / Gemini
# Date: May 6, 2025
# Description: Manages tool registry, execution, and SCHEMA definitions for Rix V52.

print(f"--- Loading Module: {__name__} (V2.3.0 - V52 Imports) ---", flush=True)

import inspect
import logging
import sys
import importlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple

# --- Rix Core Module Imports (V52 Structure) ---
try:
    from Rix_Brain.core import config_manager as rix_config # Alias
    # Potentially need RixState if memory/sciic managers are passed via state later
    # from Rix_Brain.core import global_state as RixState
    RIX_BRAIN_LOADED = True # Assume if config imports, core is available
    print(f"--- {__name__}: Rix Brain core imports successful.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Import Fail: {e}", flush=True)
    RIX_BRAIN_LOADED = False
    rix_config = None # Set to None to prevent errors below

# --- Logging Setup ---
logger = logging.getLogger("Rix_Heart") # Use central logger name

# --- Global Variables & State ---
FUNCTION_REGISTRY: Dict[str, Callable] = {}
SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {}
# These are now intended to be passed during initialization if needed, not directly from RixState here.
_memory_manager_instance: Optional[Any] = None
_sciic_manager_instance: Optional[Any] = None
_tools_initialized: bool = False
TOOL_IMPLEMENTATIONS_DIR: Optional[Path] = None # Set during initialization

def initialize(memory_manager: Optional[Any] = None, sciic_manager: Optional[Any] = None) -> bool:
    """
    Initializes the Rix Tool System, loading tools and schemas.
    Pass references to memory/sciic managers if tools require them.
    """
    global _memory_manager_instance, _sciic_manager_instance, FUNCTION_REGISTRY, SCHEMA_REGISTRY
    global _tools_initialized, TOOL_IMPLEMENTATIONS_DIR
    if _tools_initialized:
        logger.warning(f"{__name__}: Tool system already initialized. Re-initializing.")
    else:
        logger.info(f"{__name__}: Initializing Rix Tool System (V2.3.0 - V52 Structure)...")

    _memory_manager_instance = memory_manager
    # logger.info(f"Stored Mem Mgr ref type: {type(_memory_manager_instance).__name__}") # Keep if needed
    _sciic_manager_instance = sciic_manager
    # logger.info(f"Stored SCIIC Mgr ref type: {type(_sciic_manager_instance).__name__}") # Keep if needed

    FUNCTION_REGISTRY.clear()
    logger.debug(f"{__name__}: Cleared function registry.")
    SCHEMA_REGISTRY.clear()
    logger.debug(f"{__name__}: Cleared schema registry.")

    # Determine tools directory path
    if RIX_BRAIN_LOADED and rix_config:
         try:
             # Attempt to find Rix_Brain relative to this file
             try:
                 current_dir = Path(__file__).parent.resolve() # Rix_Brain/tools
                 brain_dir = current_dir.parent # Rix_Brain
             except NameError:
                  brain_dir = Path("Rix_Brain").resolve() # Fallback

             # New path for tool implementations
             TOOL_IMPLEMENTATIONS_DIR = brain_dir / "tools" / "implementations"
             logger.info(f"{__name__}: Set Tool Implementations Directory: {TOOL_IMPLEMENTATIONS_DIR}")
         except Exception as path_e:
              logger.error(f"{__name__}: Error determining tool path: {path_e}. Tool loading will likely fail.")
              TOOL_IMPLEMENTATIONS_DIR = None
    else:
        logger.error(f"{__name__}: Cannot determine tool path: Rix Config unavailable.")
        TOOL_IMPLEMENTATIONS_DIR = None


    success = _load_tools_and_schemas() # Renamed internal function
    if success:
        logger.info(f"{__name__}: Tool/Schema loading complete. {len(FUNCTION_REGISTRY)} funcs / {len(SCHEMA_REGISTRY)} schemas registered.")
    else:
        logger.error(f"{__name__}: Tool/Schema loading failed.")

    _tools_initialized = success
    return success

def _load_tools_and_schemas() -> bool:
    """Loads all tools and their schemas from the implementations directory."""
    global FUNCTION_REGISTRY, SCHEMA_REGISTRY
    if not TOOL_IMPLEMENTATIONS_DIR or not TOOL_IMPLEMENTATIONS_DIR.is_dir():
        logger.error(f"{__name__}: Tool implementations directory is invalid or not set: {TOOL_IMPLEMENTATIONS_DIR}")
        return False

    logger.info(f"{__name__}: Registering tools/schemas from: {TOOL_IMPLEMENTATIONS_DIR}")

    # Add the parent of Rix_Brain to sys.path to allow relative imports within Rix_Brain
    # This assumes a standard project layout where Rix_Brain is importable.
    try:
         brain_dir = TOOL_IMPLEMENTATIONS_DIR.parent.parent # Should be Rix_Brain
         project_root = brain_dir.parent # Should be Project_Rix
         if str(project_root) not in sys.path:
              sys.path.insert(0, str(project_root))
              logger.debug(f"{__name__}: Temporarily added {project_root} to sys.path for tool loading.")
              path_added = True
         else:
              path_added = False
    except Exception as e:
        logger.error(f"{__name__}: Failed to adjust sys.path for tool loading: {e}")
        return False # Cannot proceed reliably without correct path

    reg_ok = True
    tool_count = 0
    schema_count = 0

    for fp in TOOL_IMPLEMENTATIONS_DIR.glob("*.py"):
        if fp.name == "__init__.py":
            continue
        # Construct the module import path relative to the project root added to sys.path
        # Example: Rix_Brain.tools.implementations.basic_io
        mod_name = f"Rix_Brain.tools.implementations.{fp.stem}"
        try:
            logger.debug(f"{__name__}: Attempting to import tool module: {mod_name}")
            mod = importlib.import_module(mod_name)
            # Optional: Reload if already loaded (useful during development)
            # if mod_name in sys.modules:
            #     mod = importlib.reload(mod)

            # Register Functions
            for fname, fobj in inspect.getmembers(mod, inspect.isfunction):
                # Check if function is defined directly in this module (not imported)
                # and doesn't start with an underscore (private convention)
                if not fname.startswith("_") and inspect.getmodule(fobj) == mod:
                    if fname in FUNCTION_REGISTRY:
                        logger.warning(f"{__name__}: Function name collision for '{fname}'. Overwriting.")
                    FUNCTION_REGISTRY[fname] = fobj
                    logger.debug(f"{__name__}: Registered function: '{fname}'")
                    tool_count += 1

            # Register Schemas (Look for TOOL_SCHEMAS dictionary in the module)
            mod_schemas = getattr(mod, 'TOOL_SCHEMAS', None)
            if isinstance(mod_schemas, dict):
                 for sname, sdef in mod_schemas.items():
                     if sname in SCHEMA_REGISTRY:
                         logger.warning(f"{__name__}: Schema name collision for '{sname}'. Overwriting.")
                     # Basic schema validation
                     if isinstance(sdef, dict) and all(k in sdef for k in ['name', 'description', 'parameters']):
                         if sdef['name'] == sname: # Ensure schema name matches dict key
                             SCHEMA_REGISTRY[sname] = sdef
                             logger.debug(f"{__name__}: Registered schema: '{sname}'")
                             schema_count += 1
                         else:
                              logger.warning(f"{__name__}: Schema key '{sname}' != schema name '{sdef['name']}' in {mod_name}. Skipping.")
                     else:
                         logger.warning(f"{__name__}: Invalid schema format for '{sname}' in {mod_name}. Skipping.")
            elif mod_schemas is not None:
                logger.warning(f"{__name__}: Found 'TOOL_SCHEMAS' in {mod_name}, but it's not a dictionary. Skipping schema loading.")

        except ModuleNotFoundError:
            logger.error(f"{__name__}: Could not import tool module '{mod_name}'. Check file path and dependencies.", exc_info=True)
            reg_ok = False
        except Exception as e:
            logger.error(f"{__name__}: Error processing tool module {mod_name}: {e}", exc_info=True)
            reg_ok = False

    # Clean up sys.path if modified
    if path_added:
        try:
            sys.path.remove(str(project_root))
            logger.debug(f"{__name__}: Removed {project_root} from sys.path.")
        except ValueError:
            pass # Path wasn't there to remove

    logger.info(f"{__name__}: Tool/Schema scan finished. Registered {tool_count} functions, {schema_count} schemas.")
    return reg_ok

def get_available_tools() -> List[str]:
    """Returns a sorted list of names of registered tool functions."""
    return sorted(list(FUNCTION_REGISTRY.keys()))

def get_tool_schemas(tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Returns a list of tool schemas (for function calling).
    If tool_names is None, returns all registered schemas.
    Otherwise, returns schemas only for the specified tool names.
    """
    if not SCHEMA_REGISTRY:
        logger.warning(f"{__name__}.get_tool_schemas: No schemas registered.")
        return []
    if tool_names is None:
        # Return all registered schemas
        return list(SCHEMA_REGISTRY.values())
    else:
        # Filter schemas based on the provided names
        schemas: List[Dict[str, Any]] = []
        requested_names = set(tool_names)
        found_names = set()
        for name, schema in SCHEMA_REGISTRY.items():
            if name in requested_names:
                schemas.append(schema)
                found_names.add(name)
        # Log warnings for requested schemas that were not found
        missing_names = requested_names - found_names
        if missing_names:
             logger.warning(f"{__name__}.get_tool_schemas: Schemas not found for: {', '.join(missing_names)}")
        return schemas

def execute_tool(tool_name: str, tool_args: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes a registered tool function with the given arguments.
    Injects memory_manager and sciic_manager if the tool requests them.
    """
    log_prefix = f"[{session_id or 'NoSID'}] ToolExec"
    logger.info(f"{log_prefix}: Attempting execution - Tool='{tool_name}', Args={tool_args}")

    # Default error response structure
    default_err = {
        "command": tool_name, "status": "failed", "result": None,
        "error": {"code": "ERR_TOOL_EXEC_GENERIC", "message": "Tool execution failed."},
        "message": f"Failed to execute tool '{tool_name}'.",
        "metadata": {"input_args": tool_args}
    }

    if not _tools_initialized:
        logger.error(f"{log_prefix}: Tool system not initialized.")
        default_err["error"]["message"] = "Tool system not ready."
        default_err["error"]["code"] = "ERR_TOOL_SYS_NOT_INIT"
        return default_err

    if tool_name not in FUNCTION_REGISTRY:
        logger.error(f"{log_prefix}: Tool '{tool_name}' not found in registry.")
        default_err["error"]["code"] = "ERR_TOOL_UNKNOWN"
        default_err["error"]["message"] = f"Tool function '{tool_name}' is not registered."
        default_err["message"] = f"Unknown tool: '{tool_name}'."
        return default_err

    func = FUNCTION_REGISTRY[tool_name]
    try:
        sig = inspect.signature(func)
        params = sig.parameters
        final_args: Dict[str, Any] = {}
        missing_required: List[str] = []
        provided_args_keys = set(tool_args.keys())
        valid_param_names = set(params.keys())

        # Inject special context arguments if function signature asks for them
        if 'memory_manager' in params and _memory_manager_instance:
            final_args['memory_manager'] = _memory_manager_instance
        if 'sciic_manager' in params and _sciic_manager_instance:
            final_args['sciic_manager'] = _sciic_manager_instance
        if 'session_id' in params and session_id is not None:
            final_args['session_id'] = session_id

        # Process arguments provided by the caller (LLM)
        for pname, pdetail in params.items():
            # Skip already injected special params
            if pname in final_args:
                continue

            if pname in tool_args:
                final_args[pname] = tool_args[pname]
                provided_args_keys.discard(pname) # Mark as used
            elif pdetail.default is inspect.Parameter.empty:
                # Parameter is required but not provided
                missing_required.append(pname)

        # Check for missing required arguments
        if missing_required:
            msg = f"Missing required arguments for tool '{tool_name}': {', '.join(missing_required)}"
            logger.error(f"{log_prefix}: {msg}")
            default_err["error"]["code"] = "ERR_TOOL_MISSING_ARGS"
            default_err["error"]["message"] = msg
            default_err["message"] = msg
            return default_err

        # Log any unexpected arguments provided by the LLM
        if provided_args_keys:
            logger.warning(f"{log_prefix}: Ignoring unexpected arguments provided for tool '{tool_name}': {', '.join(provided_args_keys)}")
            default_err["metadata"]["ignored_args"] = list(provided_args_keys)

        # Execute the function
        logger.debug(f"{log_prefix}: Calling tool '{tool_name}' with final arg keys: {list(final_args.keys())}")
        result_dict = func(**final_args)
        logger.info(f"{log_prefix}: Tool '{tool_name}' finished execution.")

        # Validate and standardize the result format
        if not isinstance(result_dict, dict):
            msg = f"Tool '{tool_name}' returned invalid type '{type(result_dict).__name__}'. Expected dict."
            logger.error(f"{log_prefix}: {msg}")
            default_err["error"]["code"] = "ERR_TOOL_BAD_RETURN_TYPE"
            default_err["error"]["message"] = msg
            default_err["message"] = msg
            return default_err
        else:
            logger.info(f"{log_prefix}: Tool '{tool_name}' result status: '{result_dict.get('status', '[N/A]')}'")
            # Ensure standard keys exist
            final_result = {
                "command": tool_name,
                "status": result_dict.get("status", "unknown").lower(),
                "result": result_dict.get("result"), # The actual data payload
                "error": result_dict.get("error"), # Error details if status is failed
                "message": result_dict.get("message", ""), # User-friendly message
                "metadata": result_dict.get("metadata", {}) # Additional info
            }
            # Merge original input args and ignored args into final metadata
            final_result["metadata"]["input_args"] = tool_args
            if "ignored_args" in default_err["metadata"]:
                 final_result["metadata"]["ignored_args"] = default_err["metadata"]["ignored_args"]
            return final_result

    except TypeError as te:
        # Catches errors if LLM provided args with wrong types or func signature mismatch
        logger.exception(f"{log_prefix}: TypeError executing tool '{tool_name}': {te}")
        default_err["error"]["code"] = "ERR_TOOL_TYPE_MISMATCH"
        default_err["error"]["message"] = f"Argument type or signature mismatch: {te}"
        default_err["message"] = f"Internal argument error calling tool '{tool_name}'."
        return default_err
    except Exception as e:
        logger.exception(f"{log_prefix}: Unexpected error executing tool '{tool_name}': {e}")
        default_err["error"]["code"] = "ERR_TOOL_UNEXPECTED"
        default_err["error"]["message"] = f"Unexpected execution error: {str(e)}"
        default_err["message"] = f"Unexpected error running tool '{tool_name}'."
        return default_err


print(f"--- Module Defined: {__name__} (V2.3.0 - V52 Imports) ---", flush=True)
# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\tools\tool_manager.py ---