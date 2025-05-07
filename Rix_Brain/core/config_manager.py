# --- START OF FILE Rix_Brain/core/config_manager.py ---
# Version: 1.5.1 (V60.2 - Env Var Root, ADC Priority)
# Author: Vishal Sharma / Gemini
# Date: May 7, 2025
# Description: Handles loading config from file/env. Prioritizes RIX_PROJECT_ROOT env var.
#              Reads GOOGLE_APPLICATION_CREDENTIALS env var but avoids forcing local keys on Cloud Run.

print(f"--- Loading Module: {__name__} (V1.5.1 - Env Var Root, ADC Priority) ---", flush=True)

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys

# --- Third-Party Imports ---
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
    print(f"--- {__name__}: python-dotenv imported.", flush=True)
except ImportError:
    DOTENV_AVAILABLE = False
    # Use basicConfig here ONLY if you are sure logging isn't configured yet by an entry point
    # logging.basicConfig(level=logging.WARNING)
    # logging.getLogger(__name__).warning(...)
    print(f"INFO ({__name__}): python-dotenv not found. Cannot load from .env files.")

# --- Logging Setup ---
# It's better if the main entry point (like run_rix_cli.py or service main.py) configures logging.
# Assume logger 'Rix_Heart' might be available. If not, basic logging might occur if libs call logging.
logger = logging.getLogger("Rix_Heart")

# --- Path Definitions (Environment Variable Priority) ---
_project_root_env = os.getenv('RIX_PROJECT_ROOT')
_env_adc_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') # Check if ADC path is already set externally

if _project_root_env:
    # If RIX_PROJECT_ROOT ENV var is set (expected in container), use it
    PROJECT_ROOT_DIR = Path(_project_root_env).resolve()
    logger.info(f"{__name__}: Using RIX_PROJECT_ROOT from environment: {PROJECT_ROOT_DIR}")
else:
    # Fallback for local development (if ENV var not set)
    logger.warning(f"{__name__}: RIX_PROJECT_ROOT env var not set. Falling back to Path(__file__) logic for local dev.")
    try:
        _CONFIG_MANAGER_FILE_PATH = Path(__file__).resolve()
        # __file__ -> config_manager.py -> core -> Rix_Brain -> PROJECT_ROOT_DIR
        RIX_BRAIN_DIR_FALLBACK = _CONFIG_MANAGER_FILE_PATH.parent.parent 
        PROJECT_ROOT_DIR = RIX_BRAIN_DIR_FALLBACK.parent
        logger.info(f"{__name__}: Fallback - Determined PROJECT_ROOT_DIR via Path: {PROJECT_ROOT_DIR}")
    except Exception as e_path_fallback:
        # Super fallback if __file__ is somehow undefined (e.g., interactive, frozen app)
        logger.error(f"{__name__}: Could not determine project root via ENV var or Path(__file__): {e_path_fallback}. Defaulting to '.'")
        PROJECT_ROOT_DIR = Path(".").resolve()

# Define other key paths relative to the determined PROJECT_ROOT_DIR
RIX_BRAIN_DIR = PROJECT_ROOT_DIR / "Rix_Brain"
RIX_AUTH_DIR = RIX_BRAIN_DIR / "Rix_Auth" # Used for local SA key fallback checks
CONFIG_FILE_PATH = PROJECT_ROOT_DIR / "config.json"
# Define local key path but don't assume it's the primary ADC source
SERVICE_ACCOUNT_KEY_FILENAME = "rixagi-8a4b40e747e5.json" # Your specific key name
SERVICE_ACCOUNT_KEY_PATH_LOCAL = RIX_AUTH_DIR / SERVICE_ACCOUNT_KEY_FILENAME

# Print resolved paths for debugging during import/startup
print(f"--- {__name__}: Final PROJECT_ROOT_DIR: {PROJECT_ROOT_DIR}", flush=True)
print(f"--- {__name__}: Final RIX_BRAIN_DIR (derived): {RIX_BRAIN_DIR}", flush=True)
print(f"--- {__name__}: Final CONFIG_FILE_PATH (derived): {CONFIG_FILE_PATH}", flush=True)
print(f"--- {__name__}: Local SA Key Path check: {SERVICE_ACCOUNT_KEY_PATH_LOCAL}", flush=True)

# --- Module Global State for Config ---
_CONFIG_DATA: Dict[str, Any] = {}
_ENV_LOADED: bool = False
_CONFIG_LOADED: bool = False
_GOOGLE_API_KEY: Optional[str] = None
# This variable now JUST reports what the OS ENV VAR is set to, or None.
# It doesn't try to determine the effective path based on file existence here.
_GOOGLE_APP_CREDENTIALS_ENV_VAR: Optional[str] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# --- Internal Loading Functions ---

def _load_environment_variables():
    """
    Loads environment variables from OS and potentially .env files.
    Determines GOOGLE_API_KEY and records GOOGLE_APPLICATION_CREDENTIALS if set in env.
    """
    global _ENV_LOADED, _GOOGLE_API_KEY, _GOOGLE_APP_CREDENTIALS_ENV_VAR
    if _ENV_LOADED: return
    logger.info(f"{__name__}: Loading environment variables...")

    # Define potential .env file locations relative to project root
    env_files_to_try = [
        PROJECT_ROOT_DIR / ".env",     # Project root .env
        RIX_AUTH_DIR / ".env",         # Rix_Auth/.env
        RIX_AUTH_DIR / "gemini.env"    # Rix_Auth/gemini.env
    ]

    if DOTENV_AVAILABLE:
        for env_path in env_files_to_try:
            if env_path.is_file():
                try:
                    logger.debug(f"Loading .env file: {env_path}")
                    # Load into OS environment, potentially overriding existing vars
                    load_dotenv(dotenv_path=env_path, override=True, verbose=False)
                except Exception as e_dotenv:
                     logger.error(f"Error loading .env file '{env_path}': {e_dotenv}", exc_info=True)
            else:
                logger.debug(f".env file not found at: {env_path}")
    else:
        logger.warning(f"{__name__}: python-dotenv not installed. Cannot load settings from .env files.")

    # Get API Key (from potentially newly loaded env vars)
    _GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if _GOOGLE_API_KEY: logger.info(f"{__name__}: GOOGLE_API_KEY found via environment/dotenv.")
    else: logger.warning(f"{__name__}: GOOGLE_API_KEY not found in environment/dotenv.")

    # Get explicit ADC path if set in environment
    _GOOGLE_APP_CREDENTIALS_ENV_VAR = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if _GOOGLE_APP_CREDENTIALS_ENV_VAR:
         logger.info(f"{__name__}: GOOGLE_APPLICATION_CREDENTIALS environment variable is SET: '{_GOOGLE_APP_CREDENTIALS_ENV_VAR}'")
         # We DO NOT set os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ourselves here.
         # We let the environment (OS, Docker ENV, Cloud Run runtime) manage this variable.
         # Initialization code will check this variable's value later via get_google_app_credentials().
    else:
         logger.info(f"{__name__}: GOOGLE_APPLICATION_CREDENTIALS environment variable is NOT SET. ADC will use other methods (gcloud auth local, runtime SA on Cloud Run, etc.).")
         # We also don't attempt to set it from the local fallback file here.

    _ENV_LOADED = True
    print(f"--- {__name__}: Environment variables loading attempted.", flush=True)


# --- START OF CORRECTED _load_config_file function in Rix_Brain/core/config_manager.py ---

def _load_config_file():
    """Loads configuration from the config.json file."""
    global _CONFIG_LOADED, _CONFIG_DATA
    if _CONFIG_LOADED: return

    logger.info(f"{__name__}: Attempting to load config file from: {CONFIG_FILE_PATH}")
    
    default_config = { 
        "MANAGER_MODEL": "gemini-2.5-flash-preview-04-17", "THINKER_MODEL": "gemini-2.5-pro-preview-03-25",
        "REFINER_MODEL": "gemini-2.5-flash-preview-04-17", "CLASSIFIER_MODEL": "gemini-2.5-pro-preview-03-25",
        "MEMORY_WRITER_MODEL": "gemini-2.5-flash-preview-04-17", "WORKER_MODEL": "gemini-2.5-flash-preview-04-17",
        "GOOGLE_API_KEY": None, "GOOGLE_CLOUD_PROJECT": None, "GOOGLE_CLOUD_LOCATION": None, 
        "MANAGER_TEMPERATURE": 1.0, "THINKER_TEMPERATURE": 1.0, "CLASSIFIER_TEMPERATURE": 1.0, "MEMORY_WRITER_TEMPERATURE": 1.0, "WORKER_TEMPERATURE": 1.0,
        "EMBEDDING_MODEL": "text-embedding-005", "FIRESTORE_DATABASE_ID": "rix-agi-chat",
        "RIX_CLASSIFIER_SERVICE_URL": None, "RIX_THINKER_SERVICE_URL": None,
        "RIX_TOOL_EXECUTOR_SERVICE_URL": None, "RIX_MEMORY_WRITER_SERVICE_URL": None,
        "RIX_MANAGER_SERVICE_URL": None, "RIX_FINALIZER_SERVICE_URL": None, 
        "MAX_ORCHESTRATION_STEPS": 12, "ACKNOWLEDGE_WORK_FLOWS": True, "ACKNOWLEDGE_ASK_FLOWS": True,
        "DB_USER": "postgres", "DB_NAME": "postgres", "DB_INSTANCE_CONNECTION_NAME": None,
        "DB_PASSWORD_SECRET_NAME": None, "DB_IAM_USER": None,
        "RIX_PROJECT_PATHS_CONFIG": { 
            "ASSUMED_PROJECT_ROOT_IN_CONTAINER": "/app", "RIX_BRAIN_SUBDIR": "Rix_Brain",
            "CONFIG_SUBDIR_FROM_PROJECT_ROOT": ".", "SOULS_SUBDIR_FROM_RIX_BRAIN": "agents/souls",
            "RIX_AUTH_SUBDIR_FROM_RIX_BRAIN": "Rix_Auth", "CARTOON_DIRECTOR_SUBDIR_FROM_PROJECT_ROOT": "rix_cartoon_director",
            "CARTOON_TOOLS_SUBDIR_FROM_CARTOON_DIRECTOR": "tools", "CARTOON_MEMORY_SUBDIR_FROM_CARTOON_DIRECTOR": "memory"
        },
        "MANAGER_SOUL_FILENAME": "manager_v3_21.json", "THINKER_SOUL_FILENAME": "thinker_v4_0.json",
        "CLASSIFIER_SOUL_FILENAME": "classifier_v1_2.json", "MEMORY_WRITER_SOUL_FILENAME": "memory_writer_v1_0.json"
    }

    # Load from file or use defaults
    if CONFIG_FILE_PATH.is_file():
        try:
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            logger.info(f"{__name__}: Configuration successfully loaded from: {CONFIG_FILE_PATH}")
            _CONFIG_DATA = {**default_config, **loaded_config} 
        except Exception as e:
            logger.error(f"{__name__}: Error loading or parsing config.json from {CONFIG_FILE_PATH}: {e}. Using defaults ONLY.", exc_info=True)
            _CONFIG_DATA = default_config
    else:
        logger.warning(f"{__name__}: Config file {CONFIG_FILE_PATH} not found. Using defaults ONLY.")
        _CONFIG_DATA = default_config

    # Ensure temperatures are floats AFTER loading
    temp_keys = [k for k in _CONFIG_DATA if k.endswith("_TEMPERATURE")]
    for key in temp_keys:
        try:
            _CONFIG_DATA[key] = float(_CONFIG_DATA.get(key))
        except (ValueError, TypeError, KeyError):
            default_temp = 0.7 
            logger.warning(f"{__name__}: Invalid/missing temp format for {key}. Setting default {default_temp}.")
            _CONFIG_DATA[key] = default_temp

    # --- SET FLAG *BEFORE* CALLING get_config ---
    _CONFIG_LOADED = True 
    # --------------------------------------------

    logger.debug(f"{__name__}: Config data loaded and processed.")
    print(f"--- {__name__}: Config file loading logic complete.", flush=True)

    # --- NOW check required keys using get_config, which will work safely ---
    # Note: get_config itself calls ensure_loaded, which now sees _CONFIG_LOADED as True
    # and won't call _load_config_file again.
    required_keys = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"] # Add any others absolutely required
    missing_keys = [k for k in required_keys if not get_config(k)] 
    if missing_keys:
         logger.critical(f"{__name__}: CRITICAL CONFIG MISSING after loading! Keys not found/set: {', '.join(missing_keys)}")
         # Optionally raise an exception here if these are fatal
         # raise ValueError(f"Missing required config keys: {missing_keys}")
    # -----------------------------------------------------------------------

# --- END OF CORRECTED _load_config_file function ---


def ensure_loaded():
    """Ensures both environment variables and config file are loaded."""
    # Load environment variables first, as they have priority in get_config
    if not _ENV_LOADED: _load_environment_variables()
    if not _CONFIG_LOADED: _load_config_file()

def get_config(key: str, default: Optional[Any] = None) -> Any:
    """
    Gets a config value, prioritizing OS environment, then config file, then default.
    Uses uppercase convention for environment variable keys.
    """
    ensure_loaded()
    # 1. Check OS Environment (using uppercase key)
    env_var_key = key.upper()
    env_value = os.getenv(env_var_key)
    if env_value is not None:
        logger.debug(f"{__name__}: Config key '{key}' found in environment (as '{env_var_key}').")
        # Basic type conversion for common env var strings
        val_lower = env_value.lower()
        if val_lower == 'true': return True
        if val_lower == 'false': return False
        if val_lower == 'none' or val_lower == 'null': return None
        try: return int(env_value)
        except ValueError: pass
        try: return float(env_value)
        except ValueError: pass
        # Return raw string if no conversion matches
        return env_value

    # 2. Check loaded config file data (_CONFIG_DATA) using original key case
    # The _CONFIG_DATA dictionary holds merged values (config file over defaults)
    if key in _CONFIG_DATA:
         # Return value from config/defaults if key exists (even if value is None)
         # unless an explicit 'default' arg was passed to this function.
         # Let's refine: return config value if key exists, otherwise the function's default.
        config_value = _CONFIG_DATA.get(key) # Using .get() avoids KeyError
        logger.debug(f"{__name__}: Config key '{key}' found in loaded config/defaults.")
        return config_value
        # Removed the check for `if config_value is not None:` to allow returning None
        # if it's explicitly set to null in config.json and no env var overrides it.

    # 3. Return the function's default argument if key not found anywhere
    logger.debug(f"{__name__}: Config key '{key}' not found in environment or config/defaults. Returning function default: {default}")
    return default

def get_google_api_key() -> Optional[str]:
    """Gets the Google API Key (primarily from environment/.env)."""
    # ensure_loaded() is called by get_config
    return get_config("GOOGLE_API_KEY") # Let get_config handle priority

def get_google_app_credentials() -> Optional[str]:
    """
    Returns the value of the GOOGLE_APPLICATION_CREDENTIALS environment variable, if set.
    Returns None otherwise. Does not check for local file existence directly.
    """
    ensure_loaded() # Ensures _GOOGLE_APP_CREDENTIALS_ENV_VAR is populated
    return _GOOGLE_APP_CREDENTIALS_ENV_VAR

# --- Utility functions using the config-based paths ---
# Note: These functions will only work correctly AFTER config is loaded (ensure_loaded is called).

def get_path_config() -> Dict[str, Any]:
    """Returns the dictionary stored under RIX_PROJECT_PATHS_CONFIG."""
    return get_config("RIX_PROJECT_PATHS_CONFIG", {}) # Return empty dict if missing

def get_project_root() -> Path:
    """Gets the project root path, prioritizing RIX_PROJECT_ROOT env var."""
    # This simply returns the globally determined PROJECT_ROOT_DIR
    # It's determined once when the module loads.
    ensure_loaded() # Ensure path determination logic has run
    return PROJECT_ROOT_DIR

def get_rix_brain_dir() -> Path:
    """Gets the resolved path to the Rix_Brain directory."""
    ensure_loaded()
    return RIX_BRAIN_DIR # Uses the globally determined variable

def get_souls_dir() -> Path:
    """Gets the resolved path to the agents/souls directory."""
    ensure_loaded()
    path_cfg = get_path_config()
    souls_subdir = path_cfg.get("SOULS_SUBDIR_FROM_RIX_BRAIN", "agents/souls")
    return get_rix_brain_dir() / souls_subdir

def get_rix_auth_dir() -> Path:
    """Gets the resolved path to the Rix_Auth directory."""
    ensure_loaded()
    path_cfg = get_path_config()
    auth_subdir = path_cfg.get("RIX_AUTH_SUBDIR_FROM_RIX_BRAIN", "Rix_Auth")
    return get_rix_brain_dir() / auth_subdir

# --- Load config on module import ---
ensure_loaded()

print(f"--- Module Defined: {__name__} (V1.5.1 - Env Var Root, ADC Priority) ---", flush=True)
# --- END OF FILE Rix_Brain/core/config_manager.py ---