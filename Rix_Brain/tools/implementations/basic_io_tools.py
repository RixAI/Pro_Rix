# --- START OF MODIFIED FILE Rix_Brain/tool_implementations/basic_io_tools.py ---
# C:\Rix_Dev\Project_Rix\Rix_Brain\tool_implementations\basic_io_tools.py
# Version: 1.0.2 (V43.1 - Add Schemas) # Version Updated
# Author: Vishal Sharma / Grok 3
# Date: May 2, 2025 # Updated Date
# Description: Tool implementations for basic file system operations. Includes Schemas.

import logging
import os
import sys
import json
import shutil
import pathlib
from typing import Dict, Optional, List, Tuple, Any

logger = logging.getLogger("Rix_Heart")

# --- Constants ---
MAX_READ_FILE_SIZE = 1024 * 1024 # 1 MB

# --- Tool Schemas (for Native Function Calling) ---
TOOL_SCHEMAS = {
    "read_file": {
        "name": "read_file",
        "description": "Reads the content of a specified file path. Limited to 1MB size. Returns text content or error.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The exact file path to read (e.g., 'Rix_Brain/config.json' or 'C:\\path\\to\\file.txt')."
                }
            },
            "required": ["path"]
        }
    },
    "write_file": {
        "name": "write_file",
        "description": "Writes string content to a specified file path, creating parent directories if needed. Overwrites existing files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The exact file path to write to (e.g., 'Rix_Projects/output.txt' or 'C:\\path\\to\\new_file.log')."
                },
                "content": {
                    "type": "string",
                    "description": "The string content to write into the file."
                }
            },
            "required": ["path", "content"]
        }
    },
    "list_files": {
        "name": "list_files",
        "description": "Lists files and subdirectories directly within a specified directory path. Defaults to '.' (current Rix Brain directory).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list (e.g., '.' or 'Rix_Projects/subfolder' or 'C:\\path\\to\\dir'). Optional, defaults to '.'"
                }
            },
            "required": []
        }
    },
     "make_dirs": {
        "name": "make_dirs",
        "description": "Creates a directory at the specified path. Also creates parent directories if they do not exist. Does nothing if the directory already exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The full directory path to create (e.g., 'Rix_Projects/new_folder/sub_folder' or 'C:\\path\\to\\new\\dir')."
                }
            },
            "required": ["path"]
        }
    },
    "delete_dir": {
        "name": "delete_dir",
        "description": "DANGEROUS! Deletes a directory and ALL of its contents recursively. Use with extreme caution. Includes a basic safety check against deleting high-level directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The exact directory path to delete recursively (e.g., 'Rix_Projects/temp_output')."
                }
            },
            "required": ["path"]
        }
    }
    # Add schemas for other tools in this file if any
}

# --- Helper Functions ---
def _create_tool_result(status: str, command: str, result: Any = None, error: Optional[Dict] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Creates a standardized tool result dictionary."""
    if metadata is None:
        metadata = {}
    if isinstance(result, BaseException):
        result = str(result) # Convert exception to string if passed as result
    if isinstance(error, BaseException):
        # If an exception object is passed directly to error
        error = {"code": "ERR_UNCAUGHT_TOOL_EXC", "message": str(error)}
    elif isinstance(error, dict) and "message" not in error:
        # If error is a dict but missing 'message', try to stringify it
        error["message"] = str(error)

    return {
        "command": command,
        "status": status.lower(),
        "result": result,
        "error": error,
        "message": metadata.get("message", ""), # Optional user-friendly message
        "metadata": metadata # Additional context
    }

def _validate_path(path_str: str, command: str, check_exists: Optional[str] = None) -> Tuple[Optional[pathlib.Path], Optional[Dict]]:
    """Validates a path string, resolves it, and optionally checks existence."""
    metadata = {"input_path": path_str}
    if not path_str or not isinstance(path_str, str):
        msg = f"Tool '{command}' requires a non-empty string path argument."
        logger.error(msg)
        return None, {"code": "ERR_PATH_INV_TYPE", "message": msg}

    try:
        # Resolve the path to an absolute path and normalize it
        req_path = pathlib.Path(path_str).resolve()
        metadata["resolved_path"] = str(req_path)
        exists = req_path.exists()

        # Perform existence checks if requested
        if check_exists == 'file':
            if not exists or not req_path.is_file():
                raise FileNotFoundError(f"File not found at resolved path: '{req_path}'")
        elif check_exists == 'dir':
            if not exists or not req_path.is_dir():
                raise FileNotFoundError(f"Directory not found at resolved path: '{req_path}'")
        elif check_exists == 'must_exist':
            if not exists:
                raise FileNotFoundError(f"Path does not exist at resolved path: '{req_path}'")

        # Path is valid according to checks
        return req_path, None

    except FileNotFoundError as e:
        logger.warning(f"Path validation failed for '{command}': {e}")
        return None, {"code": "ERR_PATH_NOT_FOUND", "message": str(e)}
    except ValueError as e: # Catches invalid path characters on Windows, etc.
        logger.warning(f"Path validation failed for '{command}' due to invalid characters or format: {e}")
        return None, {"code": "ERR_PATH_INVALID", "message": str(e)}
    except PermissionError as e:
        logger.error(f"Permission error during path validation for '{command}' on '{path_str}': {e}")
        return None, {"code": "ERR_PATH_PERMISSION", "message": str(e)}
    except Exception as e: # Catch any other unexpected errors during path resolution/validation
        logger.exception(f"Unexpected error during path validation for '{command}' on '{path_str}': {e}")
        return None, {"code": "ERR_PATH_UNEXPECTED", "message": f"Unexpected path validation error: {e}"}

# --- Tool Functions ---
def read_file(path: str) -> Dict[str, Any]:
    """Reads content from a specified file path, handling text, JSON, and binary placeholders."""
    command = "read_file"
    logger.info(f"Tool: {command} - Attempting to read path: {path}")

    # Validate path: must exist and be a file
    req_path, err = _validate_path(path, command, check_exists='file')
    if err:
        # Validation failed (not found, not a file, permissions, etc.)
        return _create_tool_result("failed", command, error=err, metadata={"path": path})

    meta = {"path": path, "abs_path": str(req_path), "type": "unknown"}
    try:
        # Check file size against the limit
        size = req_path.stat().st_size
        meta["size_bytes"] = size
        if size > MAX_READ_FILE_SIZE:
            raise ValueError(f"File size ({size} bytes) exceeds the maximum allowed limit ({MAX_READ_FILE_SIZE} bytes).")

        content: Any
        try:
            # Attempt to read as UTF-8 text first
            raw_text = req_path.read_text('utf-8')
            meta["enc"] = "utf-8"

            # Check if the file extension suggests JSON
            if req_path.suffix.lower() == '.json':
                try:
                    # Try to parse as JSON
                    content = json.loads(raw_text)
                    meta["type"] = 'json'
                except json.JSONDecodeError as json_err:
                    # If JSON parsing fails, log a warning but return the raw text
                    logger.warning(f"File '{path}' has .json extension but failed to parse: {json_err}. Returning as raw text.")
                    content = raw_text
                    meta["type"] = 'text'
            else:
                # If not a .json file, treat as plain text
                content = raw_text
                meta["type"] = 'text'

        except UnicodeDecodeError:
            # If UTF-8 decoding fails, treat it as binary and return a placeholder
            logger.warning(f"Could not decode file '{path}' as UTF-8. Treating as binary.")
            content = f"[Binary content: {size} bytes]" # Placeholder, do not return actual bytes
            meta["type"] = 'binary'
            meta["enc"] = None

        # Success case
        logger.info(f"Tool '{command}' successfully read file: {req_path}")
        meta["message"] = f"Successfully read content from file: {path}"
        return _create_tool_result("success", command, result=content, metadata=meta)

    except (ValueError, PermissionError, OSError) as e:
        # Handle specific errors like size limit, permissions during read/stat, or OS issues
        code = "ERR_READ_LIMIT" if isinstance(e, ValueError) else \
               "ERR_READ_PERM" if isinstance(e, PermissionError) else \
               "ERR_READ_OS" # General OS error during read/stat
        logger.error(f"Tool '{command}' failed for '{path}': {e}")
        return _create_tool_result("failed", command, error={"code": code, "message": str(e)}, metadata=meta)
    except Exception as e:
        # Catch any other unexpected errors during file reading
        logger.exception(f"Unexpected error in tool '{command}' while reading '{path}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_READ_UNEXP", "message": str(e)}, metadata=meta)

def write_file(path: str, content: str) -> Dict[str, Any]:
    """Writes string content to a specified file path, creating directories if needed."""
    command = "write_file"
    content_len = len(content) if isinstance(content, str) else 'N/A'
    logger.info(f"Tool: {command} - Attempting to write to path: {path}, Content Length: {content_len}")

    # Ensure content is a string
    if not isinstance(content, str):
        msg = f"Content for writing must be a string, but received type {type(content)}."
        logger.error(f"Tool '{command}' failed: {msg}")
        return _create_tool_result("failed", command, error={"code": "ERR_WRITE_BAD_TYPE", "message": msg})

    # Validate the path structure (but don't require the file itself to exist)
    req_path, err = _validate_path(path, command, check_exists=None)
    if err:
        # Path validation failed (e.g., invalid characters, permission error on parent dir)
        return _create_tool_result("failed", command, error=err, metadata={"path": path})

    meta = {"path": path, "abs_path": str(req_path), "content_len": len(content)}
    try:
        # Ensure the parent directory exists; create it if necessary
        req_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content to the file, overwriting if it exists
        bytes_written = req_path.write_text(content, encoding='utf-8')
        logger.info(f"Tool '{command}' successfully wrote {bytes_written} bytes to: {req_path}")
        meta["message"] = f"Successfully wrote content to file: {path}"
        # Return success message including bytes written
        return _create_tool_result("success", command, result=f"Successfully wrote {bytes_written} bytes to {path}", metadata=meta)

    except OSError as e:
        # Handle OS-level errors during directory creation or file writing (e.g., disk full, permissions)
        logger.exception(f"OS error during tool '{command}' operation on '{path}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_WRITE_OS", "message": f"OS error writing file: {e.strerror}"}, metadata=meta)
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"Unexpected error in tool '{command}' while writing to '{path}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_WRITE_UNEXP", "message": str(e)}, metadata=meta)

def list_files(path: str = ".") -> Dict[str, Any]:
    """Lists files and subdirectories directly within a specified directory path."""
    command = "list_files"
    # Default to current directory if path is None, empty, or only whitespace
    effective_path = path if path and path.strip() else "."
    logger.info(f"Tool: {command} - Attempting to list contents of path: '{effective_path}'")
    meta = {"input_path": effective_path}

    # Validate path: must exist and be a directory
    req_path, err = _validate_path(effective_path, command, check_exists='dir')
    if err:
        # Validation failed (not found, not a dir, permissions, etc.)
        return _create_tool_result("failed", command, error=err, metadata=meta)

    meta["abs_path"] = str(req_path)
    try:
        files: List[str] = []
        dirs: List[str] = []
        # Iterate through items in the directory
        for item in req_path.iterdir():
            if item.is_file():
                files.append(item.name)
            elif item.is_dir():
                dirs.append(item.name)
            # Links or other types are ignored

        # Sort the lists for consistent output
        files.sort()
        dirs.sort()

        result_data = {"files": files, "directories": dirs}
        logger.info(f"Tool '{command}' successfully listed contents of '{effective_path}': {len(files)} files, {len(dirs)} directories.")
        meta["message"] = f"Successfully listed contents of directory: '{effective_path}'"
        return _create_tool_result("success", command, result=result_data, metadata=meta)

    except PermissionError as e:
        # Handle permission errors specifically for listing directory contents
        logger.error(f"Permission error listing directory '{effective_path}' in tool '{command}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_LIST_PERM", "message": str(e)}, metadata=meta)
    except Exception as e:
        # Catch any other unexpected errors during directory iteration
        logger.exception(f"Unexpected error listing directory '{effective_path}' in tool '{command}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_LIST_UNEXP", "message": str(e)}, metadata=meta)

def make_dirs(path: str) -> Dict[str, Any]:
    """Creates a directory at the specified path, including parent directories if needed."""
    command = "make_dirs"
    logger.info(f"Tool: {command} - Attempting to ensure directory exists at path: {path}")

    # Validate the path structure (but don't require it to exist yet)
    req_path, err = _validate_path(path, command, check_exists=None)
    if err:
        # Path validation failed (e.g., invalid characters, permission error on parent)
        return _create_tool_result("failed", command, error=err, metadata={"path": path})

    meta = {"path": path, "abs_path": str(req_path)}
    try:
        # Check if the path already exists but is a file, which is an error for mkdir
        if req_path.exists() and not req_path.is_dir():
            msg = f"Path '{path}' (resolved to '{req_path}') already exists but is a file, not a directory."
            logger.error(f"Tool '{command}' failed: {msg}")
            return _create_tool_result("failed", command, error={"code": "ERR_MKDIR_EXISTS_NOT_DIR", "message": msg}, metadata=meta)

        # Create the directory(ies).
        # parents=True: Creates necessary parent directories.
        # exist_ok=True: Doesn't raise an error if the directory already exists.
        req_path.mkdir(parents=True, exist_ok=True)

        # Verify that the path now exists and is a directory
        if req_path.is_dir():
            logger.info(f"Tool '{command}' successfully ensured directory exists: {req_path}")
            meta["message"] = f"Directory exists or was successfully created: {path}"
            return _create_tool_result("success", command, result=f"Directory exists: {path}", metadata=meta)
        else:
            # This case should be unlikely if mkdir didn't raise an error, but check anyway
            msg = f"Failed to verify directory creation after mkdir command for path: {path}"
            logger.error(f"Tool '{command}' failed verification: {msg}")
            return _create_tool_result("failed", command, error={"code": "ERR_MKDIR_VERIFY_FAIL", "message": msg}, metadata=meta)

    except OSError as e:
        # Handle OS-level errors during directory creation (e.g., permissions, invalid name part)
        logger.exception(f"OS error during tool '{command}' operation on '{path}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_MKDIR_OS", "message": f"OS error creating directory: {e.strerror}"}, metadata=meta)
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"Unexpected error in tool '{command}' while creating directory '{path}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_MKDIR_UNEXP", "message": str(e)}, metadata=meta)

def delete_dir(path: str) -> Dict[str, Any]:
    """Deletes a directory and all its contents recursively (Use with extreme caution!)."""
    command = "delete_dir"
    logger.warning(f"Tool: {command} - DANGEROUS OPERATION: Attempting to recursively delete path: {path}")

    # Validate path: must exist and be a directory
    req_path, err = _validate_path(path, command, check_exists='dir')
    if err:
        # Validation failed (not found, not a dir, permissions, etc.)
        return _create_tool_result("failed", command, error=err, metadata={"path": path})

    meta = {"path": path, "abs_path": str(req_path)}
    try:
        # --- SAFETY CHECK ---
        # Prevent deleting directories that are too high up in the filesystem hierarchy.
        # Count the number of components in the resolved absolute path.
        # Examples:
        # C:\ -> 1 part
        # C:\Users -> 2 parts
        # C:\Users\Name -> 3 parts
        # / -> 1 part
        # /home -> 2 parts
        # /home/user -> 3 parts
        # Set a minimum depth threshold. Adjust as needed for your environment's safety.
        SAFETY_DEPTH_THRESHOLD = 3 # Requires path to be at least 4 levels deep (e.g., C:\Users\Name\Folder or /home/user/project)
        path_depth = len(req_path.parts)

        if path_depth <= SAFETY_DEPTH_THRESHOLD:
            msg = (f"Safety check failed: The resolved path '{req_path}' (from input '{path}') "
                   f"is too high-level (depth {path_depth} <= threshold {SAFETY_DEPTH_THRESHOLD}). "
                   f"Deletion operation aborted to prevent accidental data loss.")
            logger.error(msg)
            return _create_tool_result("failed", command, error={"code": "ERR_DELDIR_SAFETY", "message": msg}, metadata=meta)
        # --- END SAFETY CHECK ---

        # Log clearly before proceeding with the deletion
        logger.warning(f"Safety check passed. Proceeding with recursive deletion of directory: {req_path}")
        shutil.rmtree(req_path)

        # Verify that the directory no longer exists
        if not req_path.exists():
            logger.info(f"Tool '{command}' successfully deleted directory: {path} (resolved: {req_path})")
            meta["message"] = f"Successfully deleted directory and its contents: {path}"
            return _create_tool_result("success", command, result=f"Successfully deleted directory: {path}", metadata=meta)
        else:
            # This might happen if rmtree fails partially due to locks or permissions that arise during deletion
            msg = f"Failed to verify directory deletion after rmtree command for path: {path}. It might still partially exist."
            logger.error(f"Tool '{command}' verification failed: {msg}")
            return _create_tool_result("failed", command, error={"code": "ERR_DELDIR_VERIFY_FAIL", "message": msg}, metadata=meta)

    except OSError as e:
        # Handle OS-level errors during deletion (e.g., file in use, permissions)
        logger.exception(f"OS error during tool '{command}' operation on '{path}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_DELDIR_OS", "message": f"OS error deleting directory: {e.strerror}"}, metadata=meta)
    except Exception as e:
        # Catch any other unexpected errors
        logger.exception(f"Unexpected error in tool '{command}' while deleting directory '{path}': {e}")
        return _create_tool_result("failed", command, error={"code": "ERR_DELDIR_UNEXP", "message": str(e)}, metadata=meta)

# --- END OF MODIFIED FILE Rix_Brain/tool_implementations/basic_io_tools.py ---