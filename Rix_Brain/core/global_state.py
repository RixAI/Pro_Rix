# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\global_state.py ---
# Version: 1.1.0 (V52.0 - Updated Variable Names) - Formerly rix_global_state.py
# Purpose: Hold shared global state accessible by multiple modules without circular imports.

print(f"--- Loading Module: {__name__} (V1.1.0 - V52 Names) ---", flush=True) # Version Updated

import threading
from typing import Optional, Any

# --- Rix Initialization State ---
system_initialized: bool = False
initialization_error: Optional[str] = None

# --- Component References (Set during Initialization) ---
# Prompts loaded from agents/souls/
soul_manager_prompt: Optional[str] = None
soul_thinker_prompt: Optional[str] = None
soul_classifier_prompt: Optional[str] = None
soul_memory_writer_prompt: Optional[str] = None
# embedding_function: Optional[Any] = None # This seemed unused, replaced by vertex_embedding_model

# --- Cloud Clients & Resources (Set by core.initialization) ---
firestore_db_client: Optional[Any] = None # Renamed (Stores Firestore Client)
vertex_embedding_model: Optional[Any] = None # Renamed (Stores TextEmbeddingModel)
sql_engine_pool: Optional[Any] = None   # Renamed (Stores SQLAlchemy Engine/Pool)

# --- Obsolete Variables (Commented out/Removed) ---
# task_completion_queue = queue.Queue() # Likely obsolete in local sync model
# task_queue_lock = threading.Lock() # Likely obsolete


print(f"--- Module Defined: {__name__} (V1.1.0 - V52 Names) ---", flush=True) # Version Updated
# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\global_state.py ---