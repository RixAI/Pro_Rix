# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\exceptions.py ---
# Version: 1.1.0 (V52.0 - Moved to Core) - Formerly rix_exceptions.py
# Purpose: Define custom exception classes for Project Rix.

print(f"--- Loading Module: {__name__} (V1.1.0 - V52 Structure) ---", flush=True)

class LLMInvocationError(Exception):
    """Custom exception for errors during LLM calls."""
    def __init__(self, message, role=None, original_exception=None):
        super().__init__(message)
        self.role = role # Which agent/role encountered the error
        self.original_exception = original_exception # Optional underlying exception

    def __str__(self):
        # Provide a more informative string representation
        orig_exc_str = f" (Orig: {type(self.original_exception).__name__})" if self.original_exception else ""
        role_str = f"[{self.role}] " if self.role else ""
        return f"{role_str}{super().__str__()}{orig_exc_str}"

# --- Other Potential Custom Exceptions ---

class ToolExecutionError(Exception):
    """Custom exception for errors during tool execution."""
    def __init__(self, message, tool_name=None, original_exception=None):
        super().__init__(message)
        self.tool_name = tool_name
        self.original_exception = original_exception

    def __str__(self):
        orig_exc_str = f" (Orig: {type(self.original_exception).__name__})" if self.original_exception else ""
        tool_str = f"[Tool: {self.tool_name}] " if self.tool_name else ""
        return f"{tool_str}{super().__str__()}{orig_exc_str}"

class OrchestrationError(Exception):
    """Custom exception for errors during pipeline orchestration."""
    def __init__(self, message, step=None, original_exception=None):
        super().__init__(message)
        self.step = step # Which orchestration step failed
        self.original_exception = original_exception

    def __str__(self):
        orig_exc_str = f" (Orig: {type(self.original_exception).__name__})" if self.original_exception else ""
        step_str = f"[Step: {self.step}] " if self.step is not None else ""
        return f"{step_str}{super().__str__()}{orig_exc_str}"


# Add other custom exceptions here if needed later (e.g., ConfigError, HistoryError)


print(f"--- Module Defined: {__name__} (V1.1.0 - V52 Structure) ---", flush=True)
# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\exceptions.py ---