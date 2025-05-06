# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\utils.py ---
# Version: 1.2.0 (V53.0 / V54.0 - Added OIDC Helper)
# Author: Vishal Sharma / Gemini
# Date: May 7, 2025
# Description: Common utilities for Project Rix. Includes OIDC token helper.

print(f"--- Loading Module: {__name__} (V1.2.0 - OIDC Helper) ---", flush=True)

import json
import logging
import re
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- Rix Core (for global_state access if needed by future utils or project_id) ---
# We might need global_state if config is stored there and not directly imported
# from Rix_Brain.core import global_state as RixState # Could be used for project_id from config
from Rix_Brain.core import config_manager as rix_config # More direct for config access

# --- Google Cloud Auth Imports for OIDC ---
# These are essential for get_oidc_token to function
try:
    import google.auth.transport.requests
    import google.oauth2.id_token
    import google.auth # For google.auth.default() and exceptions
    GCP_AUTH_LIBS_AVAILABLE = True
    print(f"--- {__name__}: Google Auth libraries for OIDC loaded successfully.", flush=True)
except ImportError:
    GCP_AUTH_LIBS_AVAILABLE = False
    # Use a basic logger setup here if Rix_Heart isn't configured yet during module load
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(__name__).warning(
        f"{__name__}: google-auth libraries (google.auth, google.oauth2.id_token) not found. "
        "OIDC token generation (get_oidc_token) will fail. "
        "Please ensure 'google-auth' is installed."
    )

# --- Logging Setup ---
# Use central logger name, assuming it's configured by initialization or CLI entry point
# However, this module might be imported before full logging config, so be mindful.
logger = logging.getLogger("Rix_Brain.core.utils") # More specific logger

# --- Constants ---
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-.:]{5,60}$")

# --- Utility Functions ---

def validate_session_id(sid: Any, calling_func_name: Optional[str] = None) -> bool:
    """Validates the format of a session ID."""
    if isinstance(sid, str) and SESSION_ID_PATTERN.match(sid):
        return True
    caller_info = f" in {calling_func_name}" if calling_func_name else ""
    logger.error(f"Invalid session_id detected{caller_info}. SID: '{sid}' (Type: {type(sid).__name__}). Check pattern.")
    return False

def load_soul_prompt(soul_file_path: Path) -> Optional[str]:
    """Loads a soul prompt string or list of strings from a JSON file."""
    if not isinstance(soul_file_path, Path):
        logger.error(f"Invalid soul_file_path type: {type(soul_file_path)}. Expected Path object.")
        return None
    if not soul_file_path.is_file():
        logger.error(f"Soul file missing/invalid: {soul_file_path}")
        return None
    try:
        logger.debug(f"Loading soul prompt from: {soul_file_path}")
        content = soul_file_path.read_text(encoding='utf-8')
        data = json.loads(content)
        prompt_data = data.get("prompt")

        if isinstance(prompt_data, str) and prompt_data.strip():
            logger.info(f"Loaded prompt STRING from: {soul_file_path.name}")
            return prompt_data.strip()
        elif isinstance(prompt_data, list) and prompt_data:
            joined_prompt = "\n".join(str(p) for p in prompt_data if isinstance(p, (str, int, float))).strip()
            if joined_prompt:
                logger.info(f"Loaded prompt LIST (joined) from: {soul_file_path.name}")
                return joined_prompt
            else:
                logger.error(f"Prompt list in soul file empty/invalid: {soul_file_path.name}")
                return None
        else:
            logger.error(f"Prompt missing/empty/invalid type ({type(prompt_data).__name__}) in soul file: {soul_file_path.name}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding soul JSON '{soul_file_path.name}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading soul file '{soul_file_path.name}': {e}", exc_info=True)
        return None

def get_current_timestamp() -> str:
    """Returns the current UTC timestamp in ISO format with 'Z'."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds') + 'Z'

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    # ... (keep your existing parse_tool_call implementation) ...
    if not isinstance(text, str): return None
    log_prefix = f"{__name__}.parse_tool_call"; logger.debug(f"{log_prefix}: Parsing: '{text[:150]}...'")
    json_str_to_parse = None
    json_match_fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if json_match_fenced: json_str_to_parse = json_match_fenced.group(1).strip(); logger.debug(f"{log_prefix}: Found fenced JSON.")
    else:
        marker = "TOOL_CALL:"; marker_pos = text.find(marker)
        if marker_pos != -1:
            json_part = text[marker_pos + len(marker):].strip(); brace_pos = json_part.find('{')
            if brace_pos != -1:
                open_braces = 0; end_pos = -1; potential_json = json_part[brace_pos:]
                for i, char in enumerate(potential_json):
                    if char == '{': open_braces += 1
                    elif char == '}': open_braces -= 1
                    if open_braces == 0 and char == '}': end_pos = brace_pos + i + 1; break
                if end_pos != -1: json_str_to_parse = json_part[brace_pos:end_pos]
                else: logger.warning(f"{log_prefix}: Found '{marker}' and '{{' but no matching '}}'.")
            else: logger.warning(f"{log_prefix}: Found '{marker}' but no subsequent '{{'.")
        else: logger.debug(f"{log_prefix}: No TOOL_CALL marker or fenced JSON."); return None
    if not json_str_to_parse: logger.warning(f"{log_prefix}: Failed to extract JSON string."); return None
    json_str_cleaned = re.sub(r",\s*([\}\]])", r"\1", json_str_to_parse)
    try:
        parsed_json = json.loads(json_str_cleaned)
        tool_call_dict: Optional[Dict] = None
        if isinstance(parsed_json, list) and len(parsed_json) == 1 and isinstance(parsed_json[0], dict): tool_call_dict = parsed_json[0]
        elif isinstance(parsed_json, dict): tool_call_dict = parsed_json
        else: logger.warning(f"{log_prefix}: Parsed JSON not dict or single-item list. Type: {type(parsed_json)}"); return None
        if ('name' in tool_call_dict and isinstance(tool_call_dict.get('name'), str) and 'args' in tool_call_dict and isinstance(tool_call_dict.get('args'), dict)):
            logger.info(f"{log_prefix}: Successfully parsed TOOL_CALL for tool: '{tool_call_dict['name']}'"); return tool_call_dict
        else: logger.warning(f"{log_prefix}: Parsed JSON lacks valid 'name' or 'args'."); return None
    except json.JSONDecodeError as e: logger.error(f"{log_prefix}: JSON parsing failed: {e}. Text: '{json_str_cleaned[:100]}...'"); return None
    except Exception as e: logger.error(f"{log_prefix}: Unexpected error parsing TOOL_CALL: {e}", exc_info=True); return None


def format_history_snippet(history_list: List[Dict]) -> List[str]:
    # ... (keep your existing format_history_snippet implementation) ...
    if not history_list: return []
    formatted = []
    for m in history_list:
        try:
            actor = m.get('actor', '?'); name = m.get('name', actor)[:15]; msg_type = m.get('message_type', 'unknown'); content = m.get('content', '')
            if isinstance(content, (dict, list)):
                 try: content_str = json.dumps(content)
                 except Exception: content_str = f"[{type(content).__name__} content]"
            else: content_str = str(content)
            content_display = content_str[:100] + "..." if len(content_str) > 100 else content_str
            line = f"[{actor}/{name}/{msg_type}]: {content_display}"
            if msg_type == "tool_action_request":
                tool_name_val = "?"; fcd = m.get('function_call_data')
                if isinstance(fcd, list) and fcd: tool_name_val = fcd[0].get('name', '?')
                elif m.get('tool_name'): tool_name_val = m.get('tool_name')
                line = f"[{actor}/{name}/TOOL_REQ:{tool_name_val}]: {content_display}"
            elif msg_type == "tool_action_result":
                tool_name_val = m.get('tool_name') or (m.get('function_response_data',{}).get('name') if m.get('function_response_data') else '?')
                status = m.get('tool_status') or (json.loads(content_str).get('status') if isinstance(content_str, str) and content_str.startswith('{') else '?')
                line = f"[{actor}/{name}/TOOL_RES:{tool_name_val}/{status}]: {content_display}"
            formatted.append(line)
        except Exception as format_e: logger.warning(f"Error formatting history entry: {m}. Error: {format_e}"); formatted.append("[Error formatting hist entry]")
    return formatted

# --- OIDC Token Helper Functions ---
def get_google_cloud_project_id() -> Optional[str]:
    """
    Retrieves the Google Cloud Project ID.
    Tries to get it from rix_config, then from google.auth.default().
    """
    if rix_config:
        project_id_from_config = rix_config.get_config('GOOGLE_CLOUD_PROJECT')
        if project_id_from_config:
            logger.debug(f"Found GOOGLE_CLOUD_PROJECT='{project_id_from_config}' via rix_config.")
            return project_id_from_config
    
    if GCP_AUTH_LIBS_AVAILABLE and google.auth:
        try:
            _, project_id_from_adc = google.auth.default()
            if project_id_from_adc:
                logger.debug(f"Found project ID='{project_id_from_adc}' via google.auth.default().")
                return project_id_from_adc
        except google.auth.exceptions.DefaultCredentialsError:
            logger.warning("Could not determine project ID using google.auth.default(). Ensure ADC are set if config value is missing.")
    else:
        logger.warning("google-auth library not available for project ID lookup via ADC.")
    
    logger.error("Google Cloud Project ID could not be determined from config or ADC.")
    return None

def get_oidc_token(target_audience_url: str) -> Optional[str]:
    """
    Generates an OIDC ID token for a given target audience URL (e.g., Cloud Run service URL).
    This function uses Application Default Credentials (ADC). Ensure ADC are configured
    (e.g., by running `gcloud auth application-default login` in your local environment,
    or via the service account attached to a Cloud Run/Compute Engine instance).

    Args:
        target_audience_url: The URL of the Cloud Run service (or other OIDC-protected service)
                             you intend to call. This URL is used as the 'audience' (aud) claim
                             in the generated ID token.

    Returns:
        The OIDC ID token as a string if successful, otherwise None.
    """
    if not GCP_AUTH_LIBS_AVAILABLE:
        logger.error("Cannot fetch OIDC token: google-auth libraries (google.auth, google.oauth2.id_token) are not available. Please install 'google-auth'.")
        return None

    if not target_audience_url:
        logger.error("Target audience URL is required to fetch an OIDC token but was not provided.")
        return None

    try:
        logger.debug(f"Attempting to fetch OIDC ID token for audience: {target_audience_url} using Application Default Credentials.")
        # Create an credentials object using ADC.
        # The `google.auth.default()` method finds the credentials from the environment.
        # Common sources are:
        # 1. GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to a service account key file.
        # 2. gcloud CLI's well-known path for user credentials (`gcloud auth application-default login`).
        # 3. Attached service account on GCP compute environments (GCE, Cloud Run, GKE, Cloud Functions).
        creds, project = google.auth.default(scopes=['openid', 'email', 'profile']) # Scopes might not be strictly needed for fetch_id_token but good practice

        # Create a transport request object.
        # Using google.auth.transport.requests.Request() for synchronous contexts.
        # If in an async context, you might use google.auth.transport.aiohttp_client.AIOSession().
        auth_req = google.auth.transport.requests.Request()

        # Refresh credentials if they are not already fresh or don't have an id_token.
        # This step ensures that the credentials object has the necessary information
        # to request an ID token. For user credentials, it might involve a refresh token flow.
        # For service accounts, it's usually more direct.
        if not creds.valid or not hasattr(creds, 'id_token'):
             logger.debug("Refreshing credentials before fetching ID token.")
             creds.refresh(auth_req)

        # Fetch the OIDC ID token.
        # The `target_audience` is the crucial parameter here.
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, target_audience_url)
        
        logger.info(f"Successfully fetched OIDC ID token for audience: {target_audience_url}")
        return id_token

    except google.auth.exceptions.RefreshError as re:
        logger.error(f"Error refreshing credentials while trying to fetch OIDC token for {target_audience_url}: {re}", exc_info=True)
        logger.error("This can happen if ADC are misconfigured, expired, or lack permissions. Try `gcloud auth application-default login`.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching OIDC token for {target_audience_url}: {type(e).__name__} - {e}", exc_info=True)
        return None

print(f"--- Module Defined: {__name__} (V1.2.0 - OIDC Helper) ---", flush=True)

# Example Usage (for standalone testing of this module if needed)
if __name__ == '__main__':
    # Basic logging for standalone test
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] - %(message)s')
    logger.info("Running utils.py standalone test...")

    # For this test to work, Application Default Credentials must be set up.
    # e.g., run `gcloud auth application-default login` in your terminal.

    # Attempt to get project ID
    project_id = get_google_cloud_project_id()
    if project_id:
        logger.info(f"Determined Project ID: {project_id}")
    else:
        logger.warning("Could not determine Project ID for this test.")

    # You need to deploy a Cloud Run service first to get a real URL.
    # For now, you can use a placeholder. The token generation will still work
    # if your ADC are configured, but the token won't be useful until the service exists.
    # This placeholder must be an HTTPS URL.
    placeholder_cloud_run_url = "https://your-service-name-placeholder.a.run.app" 
    
    logger.info(f"Attempting to generate OIDC token for: {placeholder_cloud_run_url}")
    token = get_oidc_token(placeholder_cloud_run_url)

    if token:
        logger.info(f"Successfully generated OIDC token (first 30 chars): {token[:30]}...")
        logger.info("You can decode this token at jwt.io to inspect its 'aud' (audience) claim, "
                    f"which should match '{placeholder_cloud_run_url}'.")
    else:
        logger.error("Failed to generate OIDC token. Check ADC setup and previous error messages.")
    logger.info("utils.py standalone test complete.")

# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\utils.py ---