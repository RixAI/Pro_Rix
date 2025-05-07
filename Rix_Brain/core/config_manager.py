# --- START OF FILE Rix_Brain/core/config_manager.py ---
# Version: 1.5.0 (V60.1 - Env Var for Project Root, Cloud Run SA Friendly)
# Author: Vishal Sharma / Gemini
# Date: May 7, 2025
# Description: Handles loading config from file/env. Prioritizes RIX_PROJECT_ROOT env var.

print(f"--- Loading Module: {__name__} (V1.5.0 - Env Var Root) ---", flush=True)

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
    print(f"INFO ({__name__}): python-dotenv not found. Cannot load from .env files.")

# --- Logging Setup ---
logger = logging.getLogger("Rix_Heart") # Assuming Rix_Heart is configured elsewhere early

# --- Path Definitions (Environment Variable Priority) ---
_project_root_env = os.getenv('RIX_PROJECT_ROOT')
_service_account_key_env = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') # Check if ADC is already set externally

if _project_root_env:
    PROJECT_ROOT_DIR = Path(_project_root_env).resolve()
    logger.info(f"{__name__}: Using RIX_PROJECT_ROOT from environment: {PROJECT_ROOT_DIR}")
else:
    logger.warning(f"{__name__}: RIX_PROJECT_ROOT env var not set. Falling back to Path(__file__) logic for local dev.")
    try:
        _CONFIG_MANAGER_FILE_PATH = Path(__file__).resolve()
        RIX_BRAIN_DIR_FALLBACK = _CONFIG_MANAGER_FILE_PATH.parent.parent # core -> Rix_Brain
        PROJECT_ROOT_DIR = RIX_BRAIN_DIR_FALLBACK.parent
        logger.info(f"{__name__}: Fallback - Determined PROJECT_ROOT_DIR via Path: {PROJECT_ROOT_DIR}")
    except Exception as e_path_fallback:
        logger.error(f"{__name__}: Error determining project root via Path(__file__): {e_path_fallback}. Defaulting to '.'")
        PROJECT_ROOT_DIR = Path(".").resolve()

RIX_BRAIN_DIR = PROJECT_ROOT_DIR / "Rix_Brain"
RIX_AUTH_DIR = RIX_BRAIN_DIR / "Rix_Auth" # Used for local SA key fallback
CONFIG_FILE_PATH = PROJECT_ROOT_DIR / "config.json"
SERVICE_ACCOUNT_KEY_FILENAME = "rixagi-8a4b40e747e5.json" # Your specified key from before
SERVICE_ACCOUNT_KEY_PATH_LOCAL_FALLBACK = RIX_AUTH_DIR / SERVICE_ACCOUNT_KEY_FILENAME

print(f"--- {__name__}: Final PROJECT_ROOT_DIR: {PROJECT_ROOT_DIR}", flush=True)
print(f"--- {__name__}: Final RIX_BRAIN_DIR (derived): {RIX_BRAIN_DIR}", flush=True)
print(f"--- {__name__}: Final CONFIG_FILE_PATH (derived): {CONFIG_FILE_PATH}", flush=True)


_CONFIG_DATA: Dict[str, Any] = {}
_ENV_LOADED: bool = False
_CONFIG_LOADED: bool = False
_GOOGLE_API_KEY: Optional[str] = None
_GOOGLE_APP_CREDENTIALS_PATH_EFFECTIVE: Optional[str] = None # Path to SA key if used

def _load_environment_variables():
    global _ENV_LOADED, _GOOGLE_API_KEY, _GOOGLE_APP_CREDENTIALS_PATH_EFFECTIVE
    if _ENV_LOADED: return

    env_files_to_try = [
        PROJECT_ROOT_DIR / ".env", # Project root .env
        RIX_AUTH_DIR / ".env",     # Rix_Auth/.env
        RIX_AUTH_DIR / "gemini.env" # Rix_Auth/gemini.env
    ]

    if DOTENV_AVAILABLE:
        for env_path in env_files_to_try:
            if env_path.is_file():
                logger.debug(f"Loading .env file: {env_path}")
                load_dotenv(dotenv_path=env_path, override=True, verbose=False)
            else:
                logger.debug(f".env file not found at: {env_path}")
    else:
        logger.warning(f"{__name__}: python-dotenv not installed. Cannot load settings from .env files.")

    _GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if _GOOGLE_API_KEY: logger.info(f"{__name__}: GOOGLE_API_KEY found via environment/dotenv.")
    else: logger.warning(f"{__name__}: GOOGLE_API_KEY not found in environment/dotenv.")

    # Determine effective GOOGLE_APPLICATION_CREDENTIALS path
    # Priority: 1. Explicitly set OS ENV var, 2. Local fallback SA Key file
    # This is primarily for LOCAL development. For Cloud Run, the runtime SA should be used (ADC).
    _env_adc_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if _env_adc_path:
        logger.info(f"{__name__}: GOOGLE_APPLICATION_CREDENTIALS found in OS environment: '{_env_adc_path}'")
        _GOOGLE_APP_CREDENTIALS_PATH_EFFECTIVE = _env_adc_path
        # Do NOT override os.environ here if it's already set by the environment (e.g. Cloud Run)
    elif SERVICE_ACCOUNT_KEY_PATH_LOCAL_FALLBACK.is_file():
        _GOOGLE_APP_CREDENTIALS_PATH_EFFECTIVE = str(SERVICE_ACCOUNT_KEY_PATH_LOCAL_FALLBACK)
        # For LOCAL runs, we might set it so ADC finds it.
        # For Cloud Run, DO NOT do this if relying on runtime SA.
        # This line is problematic for Cloud Run if it doesn't find RIX_PROJECT_ROOT and tries local fallback.
        # Best to let initialization.py handle ADC directly.
        # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = _GOOGLE_APP_CREDENTIALS_PATH_EFFECTIVE # COMMENTED OUT FOR CLOUD RUN SAFERTY
        logger.info(f"{__name__}: Using local fallback Service Account Key file: '{_GOOGLE_APP_CREDENTIALS_PATH_EFFECTIVE}'")
        logger.warning(f"{__name__}: If on Cloud Run, ensure runtime SA is configured and this local key is not unintentionally used or expected.")
    else:
        logger.warning(f"{__name__}: GOOGLE_APPLICATION_CREDENTIALS not set in OS env and local fallback SA key ('{SERVICE_ACCOUNT_KEY_FILENAME}') not found in {RIX_AUTH_DIR}. ADC may rely on gcloud auth or runtime SA.")
        _GOOGLE_APP_CREDENTIALS_PATH_EFFECTIVE = None

    _ENV_LOADED = True
    print(f"--- {__name__}: Environment variables loading attempted.", flush=True)

def _load_config_file():
    global _CONFIG_LOADED, _CONFIG_DATA
    if _CONFIG_LOADED: return

    logger.info(f"{__name__}: Attempting to load config file from: {CONFIG_FILE_PATH}")
    # Default config from your provided config.json structure
    default_config = {
        "MANAGER_MODEL": "gemini-2.5-flash-preview-04-17", "THINKER_MODEL": "gemini-2.5-pro-preview-03-25",
        "CLASSIFIER_MODEL": "gemini-2.5-pro-preview-03-25", "MEMORY_WRITER_MODEL": "gemini-2.5-flash-preview-04-17",
        "GOOGLE_CLOUD_PROJECT": "rixagi", "GOOGLE_CLOUD_LOCATION": "us-central1",
        "MANAGER_TEMPERATURE": 1.0, "THINKER_TEMPERATURE": 1.0, "CLASSIFIER_TEMPERATURE": 1.0, "MEMORY_WRITER_TEMPERATURE": 1.0,
        "EMBEDDING_MODEL": "text-embedding-005", "FIRESTORE_DATABASE_ID": "rix-agi-chat",
        # Add other keys from your config that should have defaults
    }

    if CONFIG_FILE_PATH.is_file(): # Use is_file() for Path objects
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
    
    # Ensure required keys (now project/location are in defaults, but good check)
    required_keys = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"] # Add any others that MUST exist
    missing_keys = [k for k in required_keys if not _CONFIG_DATA.get(k) and not os.getenv(k.upper())]
    if missing_keys:
         logger.critical(f"{__name__}: CRITICAL CONFIG MISSING! Keys not found in config.json or environment: {', '.join(missing_keys)}")

    # Temperature conversion
    temp_keys = [k for k in _CONFIG_DATA if k.endswith("_TEMPERATURE")]
    for key in temp_keys:
        try: _CONFIG_DATA[key] = float(_CONFIG_DATA[key])
        except: _CONFIG_DATA[key] = 0.7; logger.warning(f"Invalid temp for {key}, using 0.7")

    _CONFIG_LOADED = True
    logger.debug(f"{__name__}: Final config data loaded: Preview {_str(_CONFIG_DATA)[:200]}...")
    print(f"--- {__name__}: Config file loading attempted.", flush=True)

def ensure_loaded():
    if not _ENV_LOADED: _load_environment_variables()
    if not _CONFIG_LOADED: _load_config_file()

def get_config(key: str, default: Optional[Any] = None) -> Any:
    ensure_loaded()
    env_value = os.getenv(key.upper())
    if env_value is not None: # Env var takes precedence
        # Basic type conversion for common env var strings
        if env_value.lower() == 'true': return True
        if env_value.lower() == 'false': return False
        if env_value.lower() == 'none' or env_value.lower() == 'null': return None
        try: return int(env_value)
        except ValueError: pass
        try: return float(env_value)
        except ValueError: pass
        return env_value
    
    # Fallback to loaded _CONFIG_DATA (which includes defaults)
    return _CONFIG_DATA.get(key, default)

def get_google_api_key() -> Optional[str]:
    ensure_loaded(); return _GOOGLE_API_KEY

def get_google_app_credentials() -> Optional[str]:
    """
    Returns the EFFECTIVE path to Google Application Credentials JSON file, if one is determined.
    This is primarily for local development. On Cloud Run, this should ideally be None if using runtime SA.
    """
    ensure_loaded(); return _GOOGLE_APP_CREDENTIALS_PATH_EFFECTIVE

# Helper for logging complex dicts
def _str(obj):
    if isinstance(obj, dict):
        return json.dumps({k: v if not isinstance(v, (dict, list)) else '...' for k,v in obj.items()})
    return str(obj)

ensure_loaded() # Load on module import

print(f"--- Module Defined: {__name__} (V1.5.0 - Env Var Root) ---", flush=True)
# --- END OF FILE Rix_Brain/core/config_manager.py ---