# --- START OF FULL FILE Rix_Brain/tool_implementations/web_tools.py ---
# C:\Rix_Dev\Project_Rix\Rix_Brain\tool_implementations\web_tools.py
# Version: 1.0.5 (V43.1 - Remove Header additionalProperties) # Version Updated
# Tool implementations for web interactions. Fixes schema validation error.

print(f"--- LOADING {__name__} (V1.0.5) ---", flush=True) # Use __name__

import logging
import json
from typing import Dict, Optional, Any
from urllib.parse import urlparse

# --- Web Request Library ---
REQUESTS_AVAILABLE = False
try:
    import requests
    from requests.exceptions import RequestException, Timeout, TooManyRedirects, ConnectionError as RequestsConnectionError
    REQUESTS_AVAILABLE = True
    print("INFO (web_tools): Requests library imported.")
except ImportError:
    print("WARNING (web_tools): Requests library not found. fetch_url disabled.")

logger = logging.getLogger("Rix_Heart") # Use central logger

# --- Tool Schemas (FIXED headers schema) ---
TOOL_SCHEMAS = {
    "fetch_url": {
        "name": "fetch_url",
        "description": "Fetches content from a given web URL using HTTP GET or POST. Returns text if possible, otherwise indicates binary content or error.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch (http or https)."},
                "method": {"type": "string", "enum": ["GET", "POST"], "description": "HTTP method (Default: GET)."},
                "headers": {
                    "type": "object", # Just defining as object is usually sufficient for LLM
                    "description": "Optional: A JSON object of custom headers to send (e.g., {'Authorization': 'Bearer token', 'Accept': 'application/json'}). User-Agent is set automatically.",
                    # "additionalProperties": {"type": "string"} # <<<--- REMOVED THIS INVALID LINE ---<<<
                },
                "data": {
                    "type": "object",
                    "description": "Optional: Data to send in the request body (typically for POST). Provide as a JSON object. Plain strings may also work but objects are preferred."
                }
            },
            "required": ["url"]
        }
    }
}

# --- Helper Functions ---
# Define BaseExceptionStub safely for environments where BaseException might not be imported
BaseExceptionStub = type("BaseExceptionStub", (Exception,), {})
try:
    from builtins import BaseException
except ImportError:
    BaseException = BaseExceptionStub

def _create_tool_result(status: str, command: str, result: Any = None, error: Optional[Dict] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Creates a standardized result dictionary for tool execution."""
    metadata = metadata or {}
    if isinstance(result, BaseException):
        result = str(result)
    if isinstance(error, BaseException):
        error = {"code": "ERR_UNCAUGHT_TOOL_EXC", "message": str(error)}
    elif isinstance(error, dict) and "message" not in error:
        error["message"] = str(error)
    return {"command": command, "status": status.lower(), "result": result, "error": error, "message": metadata.get("message", ""), "metadata": metadata}

# --- Tool Function ---
# (Implementation of fetch_url remains unchanged)
FETCH_TIMEOUT = 15
MAX_FETCH_SIZE = 2 * 1024 * 1024
ALLOWED_PROTOCOLS = ("http", "https")
USER_AGENT = "RixDevGod/5.0"
def fetch_url(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, data: Optional[Any] = None) -> Dict[str, Any]:
    command = "fetch_url"
    logger.info(f"Tool:{command} - URL:{url}, Meth:{method}")
    meta = {"url": url, "method": method, "prov_hdr": headers or {}, "prov_data_type": type(data).__name__ if data else None}
    if not REQUESTS_AVAILABLE:
        return _create_tool_result("failed", command, error={"code": "ERR_FETCH_MISSING_LIB", "message": "Requests missing."}, metadata=meta)
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Inv URL struct")
        if parsed.scheme not in ALLOWED_PROTOCOLS:
            raise ValueError(f"Proto '{parsed.scheme}' !allow.")
    except ValueError as e:
        logger.error(f"Inv URL:{url}-{e}")
        return _create_tool_result("failed", command, error={"code": "ERR_FETCH_INV_URL", "message": f"Inv URL:{e}"}, metadata=meta)
    req_hdr = {"User-Agent": USER_AGENT}
    req_data = None
    if headers:
        req_hdr.update(headers)
    meta["final_req_hdr"] = req_hdr
    if data:
        if isinstance(data, (dict, list)):
            req_data = json.dumps(data)
            req_hdr.setdefault('Content-Type', 'application/json')
        else:
            req_data = str(data)
        meta["sent_data_prev"] = str(req_data)[:100] + "..."
    try:
        logger.debug(f"Send {method} req {url} (T:{FETCH_TIMEOUT}s)")
        resp = requests.request(method, url, headers=req_hdr, data=req_data, timeout=FETCH_TIMEOUT, stream=True)
        resp.raise_for_status()
        cont_len = resp.headers.get('content-length')
        content = ""
        bytes_read = 0
        if cont_len and int(cont_len) > MAX_FETCH_SIZE:
            raise ValueError(f"ContLen({cont_len}) > limit({MAX_FETCH_SIZE}).")
        cont_type = resp.headers.get('content-type', '').lower()
        is_text = 'text' in cont_type or 'json' in cont_type or 'xml' in cont_type
        for chunk in resp.iter_content(chunk_size=8192):
            bytes_read += len(chunk)
            if bytes_read > MAX_FETCH_SIZE:
                raise ValueError(f"Download > limit({MAX_FETCH_SIZE}).")
            if is_text:
                try:
                    content += chunk.decode(resp.encoding or 'utf-8', errors='replace')
                except UnicodeDecodeError:
                    is_text = False
                    content = "[Binary]"
        logger.info(f"Tool '{command}' OK {url}. Stat:{resp.status_code}, Read:{bytes_read}b.")
        if 'json' in cont_type and is_text:
            try:
                res_content = json.loads(content)
                meta["parsed_as"] = "json"
            except json.JSONDecodeError:
                logger.warning("JSON parse fail->text.")
                res_content = content
                meta["parsed_as"] = "text(json fail)"
        elif is_text:
            res_content = content
            meta["parsed_as"] = "text"
        else:
            res_content = f"[Binary:{bytes_read}b]"
            meta["parsed_as"] = "binary"
        meta["message"] = f"Fetched OK:{url}"
        meta["status_code"] = resp.status_code
        meta["resp_hdr"] = dict(resp.headers)
        return _create_tool_result("success", command, result=res_content, metadata=meta)
    except Timeout:
        logger.error(f"Timeout fetch {url}")
        return _create_tool_result("failed", command, error={"code": "ERR_FETCH_TIMEOUT", "message": "Timeout."}, metadata=meta)
    except TooManyRedirects:
        logger.error(f"Redirect loop {url}")
        return _create_tool_result("failed", command, error={"code": "ERR_FETCH_REDIR", "message": "Too many redir."}, metadata=meta)
    except RequestsConnectionError as e:
        logger.error(f"Conn err {url}:{e}")
        return _create_tool_result("failed", command, error={"code": "ERR_FETCH_CONN", "message": f"Conn err:{e}"}, metadata=meta)
    except requests.exceptions.RequestException as e:
        logger.exception(f"Req fail {url}:{e}")
        code = f"ERR_FETCH_HTTP_{e.response.status_code}" if e.response is not None else "ERR_FETCH_REQ"
        return _create_tool_result("failed", command, error={"code": code, "message": str(e)}, metadata=meta)
    except ValueError as e:
        logger.error(f"Val err fetch {url}:{e}")
        return _create_tool_result("failed", command, error={"code": "ERR_FETCH_VALUE", "message": str(e)}, metadata=meta)
    except Exception as e:
        logger.exception(f"Unexpected err tool'{command}':{e}")
        return _create_tool_result("failed", command, error={"code": "ERR_FETCH_UNEXP", "message": str(e)}, metadata=meta)
# --- END OF FULL FILE Rix_Brain/tool_implementations/web_tools.py ---