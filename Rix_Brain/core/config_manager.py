# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\config_manager.py ---
# Version: 1.4.0 (V52.0 - Updated Paths & Structure) - Formerly rix_config.py
# Author: Vishal Sharma / Gemini
# Date: May 6, 2025
# Description: Handles loading config from file/env. Sets GOOGLE_APPLICATION_CREDENTIALS.

print(f"--- Loading Module: {__name__} (V1.4.0 - V52 Structure) ---", flush=True)

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
except ImportError:
    DOTENV_AVAILABLE = False
    print(f"INFO ({__name__}): python-dotenv not found. Cannot load from .env files.")

# --- Logging Setup ---
# Use central logger name, assuming it's configured by initialization or CLI entry point
logger = logging.getLogger("Rix_Heart")

# --- Path Definitions (Relative to this file's location) ---
try:
    # C:\Rix_Dev\Pro_Rix\Rix_Brain\core\config_manager.py -> Rix_Brain
    RIX_BRAIN_DIR = Path(__file__).parent.parent.resolve()
    # Project Root is parent of Rix_Brain
    PROJECT_ROOT_DIR = RIX_BRAIN_DIR.parent
    # Auth dir relative to Rix_Brain
    RIX_AUTH_DIR = RIX_BRAIN_DIR / "Rix_Auth"
    # Config file relative to Project Root
    CONFIG_FILE_PATH = PROJECT_ROOT_DIR / "config.json"
    # Env files relative to Auth dir
    ENV_FILE_PATH_AUTH_DOTENV = RIX_AUTH_DIR / ".env"
    ENV_FILE_PATH_AUTH_GEMINI = RIX_AUTH_DIR / "gemini.env"
    # Service Account Key file relative to Auth dir
    SERVICE_ACCOUNT_KEY_FILENAME = "rixagi-8a4b40e747e5.json" # Keep this configurable maybe?
    SERVICE_ACCOUNT_KEY_PATH = RIX_AUTH_DIR / SERVICE_ACCOUNT_KEY_FILENAME

    print(f"--- {__name__}: Determined RIX_BRAIN_DIR: {RIX_BRAIN_DIR}", flush=True)
    print(f"--- {__name__}: Determined PROJECT_ROOT_DIR: {PROJECT_ROOT_DIR}", flush=True)
    print(f"--- {__name__}: Determined CONFIG_FILE_PATH: {CONFIG_FILE_PATH}", flush=True)

except NameError:
    # Fallback if __file__ is not defined (e.g., interactive session)
    logger.warning(f"{__name__}: __file__ not defined. Using fallback relative paths.")
    PROJECT_ROOT_DIR = Path(".").resolve() # Assume running from project root
    RIX_BRAIN_DIR = PROJECT_ROOT_DIR / "Rix_Brain"
    RIX_AUTH_DIR = RIX_BRAIN_DIR / "Rix_Auth"
    CONFIG_FILE_PATH = PROJECT_ROOT_DIR / "config.json"
    ENV_FILE_PATH_AUTH_DOTENV = RIX_AUTH_DIR / ".env"
    ENV_FILE_PATH_AUTH_GEMINI = RIX_AUTH_DIR / "gemini.env"
    SERVICE_ACCOUNT_KEY_FILENAME = "rixagi-8a4b40e747e5.json"
    SERVICE_ACCOUNT_KEY_PATH = RIX_AUTH_DIR / SERVICE_ACCOUNT_KEY_FILENAME


# --- Module Global State for Config ---
_CONFIG_DATA: Dict[str, Any] = {}
_ENV_LOADED: bool = False
_CONFIG_LOADED: bool = False
_GOOGLE_API_KEY: Optional[str] = None
_GOOGLE_APP_CREDENTIALS: Optional[str] = None

# --- Internal Loading Functions ---

def _load_environment_variables():
    """Loads environment variables from OS and .env files."""
    global _ENV_LOADED, _GOOGLE_API_KEY, _GOOGLE_APP_CREDENTIALS
    if _ENV_LOADED: return
    logger.info(f"{__name__}: Attempting load env vars from OS and {RIX_AUTH_DIR}")

    if DOTENV_AVAILABLE:
        try:
            # Load general .env first
            if ENV_FILE_PATH_AUTH_DOTENV.is_file():
                logger.debug(f"Loading {ENV_FILE_PATH_AUTH_DOTENV}")
                load_dotenv(dotenv_path=ENV_FILE_PATH_AUTH_DOTENV, override=True, verbose=False)
            else: logger.debug(f"{ENV_FILE_PATH_AUTH_DOTENV} not found.")
            # Load gemini specific .env (potentially overriding)
            if ENV_FILE_PATH_AUTH_GEMINI.is_file():
                logger.debug(f"Loading {ENV_FILE_PATH_AUTH_GEMINI}")
                load_dotenv(dotenv_path=ENV_FILE_PATH_AUTH_GEMINI, override=True, verbose=False)
            else: logger.debug(f"{ENV_FILE_PATH_AUTH_GEMINI} not found.")
        except Exception as env_e:
            logger.error(f"Error loading .env files: {env_e}", exc_info=True)
    else: logger.warning(f"{__name__}: python-dotenv not installed. Cannot load from .env files.")

    # Get GOOGLE_API_KEY from environment (priority) or .env
    _GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if _GOOGLE_API_KEY: logger.info(f"{__name__}: GOOGLE_API_KEY found in environment.")
    else: logger.warning(f"{__name__}: GOOGLE_API_KEY not found in environment/dotenv.")

    # Determine GOOGLE_APPLICATION_CREDENTIALS path
    # Priority: 1) Explicit file, 2) OS Environment Variable
    if SERVICE_ACCOUNT_KEY_PATH.is_file():
        _GOOGLE_APP_CREDENTIALS = str(SERVICE_ACCOUNT_KEY_PATH)
        # Set environment variable for Application Default Credentials (ADC)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = _GOOGLE_APP_CREDENTIALS
        logger.info(f"{__name__}: Service Account Key file found. Set OS env GOOGLE_APPLICATION_CREDENTIALS='{_GOOGLE_APP_CREDENTIALS}'")
    else:
        logger.warning(f"{__name__}: Service Account Key '{SERVICE_ACCOUNT_KEY_FILENAME}' not found in {RIX_AUTH_DIR}. Checking OS env...")
        _GOOGLE_APP_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if _GOOGLE_APP_CREDENTIALS:
            logger.info(f"{__name__}: Using GOOGLE_APPLICATION_CREDENTIALS from OS environment: '{_GOOGLE_APP_CREDENTIALS}'")
        else:
            logger.warning(f"{__name__}: GOOGLE_APPLICATION_CREDENTIALS not found via file or OS environment. ADC might fail or use other methods.")
            _GOOGLE_APP_CREDENTIALS = None # Explicitly None if not found

    _ENV_LOADED = True
    print(f"--- {__name__}: Environment variables loaded.", flush=True)


def _load_config_file():
    """Loads configuration from the JSON file."""
    global _CONFIG_LOADED, _CONFIG_DATA
    if _CONFIG_LOADED: return

    logger.info(f"{__name__}: Loading config file from: {CONFIG_FILE_PATH}")
    default_config = { # Define defaults here
        "MANAGER_MODEL": "models/gemini-2.5-flash-preview-04-17",
        "THINKER_MODEL": "models/gemini-2.5-pro-preview-03-25",
        "CLASSIFIER_MODEL": "models/gemini-2.5-flash-preview-04-17",
        "MEMORY_WRITER_MODEL": "models/gemini-2.5-flash-preview-04-17",
        "EMBEDDING_MODEL": "models/text-embedding-004",
        "MANAGER_TEMPERATURE": 0.7,
        "THINKER_TEMPERATURE": 0.5,
        "CLASSIFIER_TEMPERATURE": 0.1,
        "MEMORY_WRITER_TEMPERATURE": 0.4,
        "LOG_LEVEL": "INFO",
        "MAX_ORCHESTRATION_STEPS": 15,
        # Add other necessary defaults for DB, etc. if not loading from .env
        "GOOGLE_CLOUD_PROJECT": None, # MUST be set in config.json or env
        "GOOGLE_CLOUD_LOCATION": None, # MUST be set in config.json or env
        "FIRESTORE_DATABASE_ID": "rix-agi-chat",
        "DB_USER": None,
        "DB_NAME": None,
        "DB_INSTANCE_CONNECTION_NAME": None,
        "DB_PASSWORD_SECRET_NAME": None,
    }

    if CONFIG_FILE_PATH.exists():
        try:
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            logger.info(f"{__name__}: Configuration successfully loaded from: {CONFIG_FILE_PATH}")
            # Merge loaded config with defaults (loaded values override defaults)
            _CONFIG_DATA = {**default_config, **loaded_config}
        except json.JSONDecodeError as e:
            logger.error(f"{__name__}: Error decoding config.json: {e}. Using defaults ONLY.", exc_info=True)
            _CONFIG_DATA = default_config
        except Exception as e:
            logger.error(f"{__name__}: Error reading config.json: {e}. Using defaults ONLY.", exc_info=True)
            _CONFIG_DATA = default_config
    else:
        logger.warning(f"{__name__}: Config file {CONFIG_FILE_PATH} not found. Using defaults ONLY.")
        _CONFIG_DATA = default_config

    # Ensure essential keys that MUST be set are checked after loading/defaults
    required_keys = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "DB_INSTANCE_CONNECTION_NAME", "DB_PASSWORD_SECRET_NAME"]
    missing_keys = [k for k in required_keys if not _CONFIG_DATA.get(k) and not os.getenv(k.upper())] # Check config and env
    if missing_keys:
         # Log error but don't raise here, let initialization handle fatal errors
         logger.error(f"{__name__}: CRITICAL CONFIG MISSING! Keys not found in config.json or environment: {', '.join(missing_keys)}")


    # Ensure temperatures are floats
    temp_keys = [k for k in _CONFIG_DATA if k.endswith("_TEMPERATURE")]
    for key in temp_keys:
        try:
            _CONFIG_DATA[key] = float(_CONFIG_DATA.get(key)) # Use .get() for safety
        except (ValueError, TypeError, KeyError):
            default_temp = 0.7 # A generic default
            logger.warning(f"{__name__}: Invalid/missing temp format for {key}. Setting default {default_temp}.")
            _CONFIG_DATA[key] = default_temp

    _CONFIG_LOADED = True
    logger.debug(f"{__name__}: Final config data (after defaults): {_CONFIG_DATA}")
    print(f"--- {__name__}: Config file loaded.", flush=True)

# --- Accessor Functions ---

def ensure_loaded():
    """Ensures both environment variables and config file are loaded."""
    if not _ENV_LOADED:
        _load_environment_variables()
    if not _CONFIG_LOADED:
        _load_config_file()

def get_config(key: str, default: Optional[Any] = None) -> Any:
    """Gets a config value, prioritizing OS environment, then config file, then default."""
    ensure_loaded()
    # 1. Check OS Environment (case-sensitive for os.getenv)
    # Use upper case convention for environment variables
    env_value = os.getenv(key.upper())
    if env_value is not None:
        logger.debug(f"{__name__}: Config key '{key}' found in environment (as '{key.upper()}').")
        # Handle common string representations
        if isinstance(env_value, str):
            val_lower = env_value.lower()
            if val_lower == 'true': return True
            if val_lower == 'false': return False
            if val_lower == 'none' or val_lower == 'null': return None
            # Attempt numeric conversion
            try: return int(env_value)
            except ValueError: pass
            try: return float(env_value)
            except ValueError: pass
        return env_value # Return raw string if other checks fail

    # 2. Check loaded config file data (_CONFIG_DATA)
    # Use original key case for dictionary lookup
    if key in _CONFIG_DATA:
        config_value = _CONFIG_DATA[key]
        if config_value is not None: # Check if key exists and value is not None
             logger.debug(f"{__name__}: Config key '{key}' found in config.json.")
             return config_value
        else: # Key exists but value is None
             logger.debug(f"{__name__}: Config key '{key}' found in config.json but value is None. Checking default.")
             # Fall through to return default

    # 3. Return default
    logger.debug(f"{__name__}: Config key '{key}' not found in environment or config.json. Returning default: {default}")
    return default

def get_google_api_key() -> Optional[str]:
    """Gets the Google API Key (primarily from environment/.env)."""
    ensure_loaded()
    return _GOOGLE_API_KEY

def get_google_app_credentials() -> Optional[str]:
    """Gets the path to the Google Application Credentials file."""
    ensure_loaded()
    return _GOOGLE_APP_CREDENTIALS

# --- REMOVED get_project_paths() ---
# Path logic should be handled where needed, e.g., initialization getting souls dir.

# --- Ensure config is loaded on module import ---
# This ensures that just importing the module makes config available
ensure_loaded()

print(f"--- Module Defined: {__name__} (V1.4.0 - V52 Structure) ---", flush=True)
# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\config_manager.py ---