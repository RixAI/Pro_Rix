# C:\Rix_Dev\Project_Rix\Rix_Brain\tool_implementations\directory_tools.py
# Version: 1.0.0
# Author: Vishal Sharma / Grok 3
# Date: April 29, 2025
# Description: Contains tools for scanning and interacting with directories.

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("Rix_Heart") # Use central logger

def scan_directory_tree(
    path_str: str = ".", depth: int = 2, max_items_per_dir: int = 30 # Adjusted defaults
) -> Dict[str, Any]:
    """
    Scans a directory tree up to a specified depth, listing files and directories.

    Args:
        path_str: The starting directory path (defaults to current directory '.').
        depth: How many levels deep to scan (e.g., 1 means only the current directory). Defaults to 2.
        max_items_per_dir: Max number of files/dirs to list per directory to prevent huge outputs. Defaults to 30.

    Returns:
        A dictionary with the status and the scanned tree structure.
    """
    command = "scan_directory_tree"
    logger.info(f"Tool '{command}' called. Path='{path_str}', Depth={depth}, MaxItems={max_items_per_dir}")
    tool_metadata = {"input_path": path_str, "input_depth": depth, "input_max_items": max_items_per_dir}
    result_payload = {
        "command": command, "status": "failed", "result": None,
        "error": None, "message": "Scan failed.", "metadata": tool_metadata
    }

    try:
        # Validate inputs
        if not isinstance(depth, int) or depth < 0:
            raise ValueError("Depth must be a non-negative integer.")
        if not isinstance(max_items_per_dir, int) or max_items_per_dir <= 0:
            raise ValueError("Max items per directory must be a positive integer.")

        start_path = Path(path_str).resolve()
        tool_metadata["resolved_path"] = str(start_path)

        if not start_path.is_dir():
            raise FileNotFoundError(f"Directory not found: '{start_path}'")

        logger.info(f"Scanning directory tree starting at: {start_path}")
        tree: Dict[str, Dict[str, list]] = {}
        items_scanned = 0
        max_total_items_limit = 500 # Overall safety limit

        for root, dirs, files in os.walk(start_path, topdown=True, onerror=lambda e: logger.warning(f"os.walk error accessing {e.filename}: {e.strerror}")):
            current_path = Path(root)
            try:
                # Calculate relative depth from the starting path
                relative_depth = len(current_path.relative_to(start_path).parts)

                # Prune recursion if max depth reached
                if relative_depth >= depth:
                    dirs[:] = [] # Stop exploring deeper in these branches
                    continue

                # Filter hidden dirs/files and sort
                dirs[:] = sorted([d for d in dirs if not d.startswith('.')])
                files = sorted([f for f in files if not f.startswith('.')])

                # Create relative key for the dictionary
                relative_key = str(current_path.relative_to(start_path)).replace("\\", "/") if current_path != start_path else "."

                # Limit items per directory
                files_limited = files[:max_items_per_dir]
                dirs_limited = dirs[:max_items_per_dir]
                files_truncated = len(files) > max_items_per_dir
                dirs_truncated = len(dirs) > max_items_per_dir

                tree[relative_key] = {
                    "dirs": dirs_limited + ["..."] if dirs_truncated else dirs_limited,
                    "files": files_limited + ["..."] if files_truncated else files_limited,
                }

                # Check overall item limit
                items_scanned += len(dirs) + len(files)
                if items_scanned > max_total_items_limit:
                    logger.warning(f"{command}: Scan truncated. Reached max total items limit ({max_total_items_limit}).")
                    tree[relative_key]["INFO"] = "Scan truncated (max total items limit reached)"
                    break # Stop the entire walk

            except Exception as walk_proc_err:
                 logger.warning(f"Error processing directory '{root}': {walk_proc_err}")
                 continue # Skip processing this directory, but continue walk if possible

        result_payload.update(
            status="success",
            result={"scanned_tree": tree}, # Nest result for consistency
            message=f"Directory scan completed for '{start_path}'.",
        )
        tool_metadata["items_found_total"] = items_scanned
        logger.info(f"'{command}': Scan successful.")

    except FileNotFoundError as e:
        result_payload.update(message=str(e), error={"code": "ERR_DIR_NOT_FOUND", "message": str(e)})
        logger.error(f"'{command}': {e}")
    except ValueError as e:
        result_payload.update(message=str(e), error={"code": "ERR_INVALID_ARGS", "message": str(e)})
        logger.error(f"'{command}': Invalid arguments: {e}")
    except PermissionError as e:
        result_payload.update(message=f"Permission error: {e}", error={"code": "ERR_PERMISSION", "message": str(e)})
        logger.error(f"'{command}': Permission error scanning '{path_str}': {e}")
    except Exception as e:
        err_msg = f"An unexpected error occurred during directory scan: {e}"
        result_payload.update(message=err_msg, error={"code": "ERR_SCAN_UNEXPECTED", "message": err_msg})
        logger.exception(f"'{command}': {err_msg}")

    return result_payload