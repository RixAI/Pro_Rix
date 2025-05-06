# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\initialization.py ---
# Version: 1.9.4 (V53.0 - Text Embedding 768d + Dimension Check)

print(f"--- Loading Module: {__name__} (V1.9.4 - Text Embedding 768d + Dim Check) ---", flush=True)

import logging
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# --- Rix Core Module Imports (V52 Structure) ---
try:
    from . import global_state as RixState
    from . import config_manager as rix_config
    from . import utils as rix_utils # utils might be needed indirectly or by other modules
    from Rix_Brain.tools import tool_manager as Rix_tools_play # Tool manager needed for init
    RIX_BRAIN_LOADED = True
    print(f"--- {__name__}: Rix Brain core imports successful.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Import Fail: {e}", flush=True)
    RIX_BRAIN_LOADED = False
    raise # Stop initialization if core components can't be imported

# --- Required Libs for LangGraph Init & Other Clients ---
LANGGRAPH_LIBS_LOADED = False
GCP_CLIENTS_LOADED = False
DB_CONNECTOR_LOADED = False
firestore = None
aiplatform = None
vertexai = None
TextEmbeddingModel = None
SecretManagerServiceClient = None
sqlalchemy = None
pg8000 = None
Connector = None
IPTypes = None
FirestoreSaver = None

print(f"--- {__name__}: Importing GCP, DB, LangGraph Libs...", flush=True)
try:
    from langchain_google_firestore import FirestoreSaver
    from google.cloud import firestore
    LANGGRAPH_LIBS_LOADED = True
    import google.cloud.aiplatform as aiplatform
    import vertexai
    # Use TextEmbeddingModel for text-embedding-005
    from vertexai.language_models import TextEmbeddingModel
    from google.cloud.secretmanager import SecretManagerServiceClient
    GCP_CLIENTS_LOADED = True
    from google.cloud.sql.connector import Connector, IPTypes
    import sqlalchemy
    import pg8000 # Required by cloud-sql-python-connector[pg8000]
    DB_CONNECTOR_LOADED = True
    logger = logging.getLogger("Rix_Heart") # Use central logger name
    logger.info("Initialization Module: Required GCP, DB & LangGraph libraries imported.")
    print(f"--- {__name__}: Lib Imports OK.", flush=True)
except ImportError as e:
    # Log critical error and potentially re-raise if essential libs are missing
    print(f"--- {__name__}: FATAL - Lib Import Fail: {e}. Check requirements.txt.", flush=True)
    logger = logging.getLogger("Rix_Heart")
    logger.critical(f"Init CRITICAL: Failed import of essential libraries: {e}.", exc_info=True)
    # Depending on which lib failed, you might want to stop here
    # For now, we let the checks later handle it, but this is a failure point.
    # raise RuntimeError(f"Failed to import essential libraries: {e}") from e # Option to halt earlier

# --- Expected Dimension (Align with target model and pgvector_memory.py) ---
# As per documentation snippet, text-embedding-005 defaults to 768
EXPECTED_EMBEDDING_DIMENSION = 768

# --- Logging Setup ---
logger = logging.getLogger("Rix_Heart") # Use central logger name established elsewhere

# --- Initialization Function ---
_init_lock = threading.Lock() # Lock to prevent concurrent initialization runs

def initialize_core() -> bool:
    """
    Initializes Rix V53 Core for LOCAL LangGraph Execution.
    Focuses on Text Embeddings (768d) and prepares for Hybrid Cloud Run calls.
    Returns True if initialization is successful, False otherwise.
    """
    print(f"--- {__name__}: ENTERING initialize_core (V1.9.4) ---", flush=True)
    with _init_lock:
        # Check if already initialized
        if getattr(RixState, 'system_initialized', False):
            logger.warning("Rix Core already initialized. Skipping re-initialization.")
            return True

        # Reset state variables at the start of initialization attempt
        logger.info(f"--- Rix Core Initialization Start (V1.9.4 - Text Embedding {EXPECTED_EMBEDDING_DIMENSION}d) ---")
        print(f"--- {__name__}: Resetting RixState...", flush=True)
        RixState.system_initialized = False
        RixState.initialization_error = "Initialization started."
        RixState.firestore_db_client = None
        RixState.vertex_embedding_model = None
        RixState.sql_engine_pool = None
        RixState.firestore_checkpointer = None
        # Reset any other state vars you might add later

        connector: Optional[Connector] = None # Define connector here for potential cleanup in finally block

        try:
            # --- Step 1: Load Config & Setup Logging ---
            print(f"--- {__name__}: STEP 1 - Config & Logging ---", flush=True)
            logger.info("Step 1: Loading Configuration and Setting up Logging...")
            if not RIX_BRAIN_LOADED or not rix_config:
                 raise RuntimeError("Rix Config Manager (core/config_manager.py) not loaded or failed import.")
            # Ensure config is loaded (it usually loads on import, but good practice)
            rix_config.ensure_loaded()
            log_level_str = rix_config.get_config("LOG_LEVEL", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            # Configure the root logger or specific loggers as needed
            logging.getLogger("Rix_Heart").setLevel(log_level) # Central Rix logger
            logging.getLogger("Rix_Brain").setLevel(log_level) # Parent for submodules
            # Set level for CLI logger if it exists (it might be configured in run_rix_cli.py)
            try: logging.getLogger("RixCli").setLevel(log_level)
            except Exception: pass # Ignore if CLI logger isn't configured yet
            logger.info(f"Logging level set to: {log_level_str}")
            print(f"--- {__name__}: STEP 1 Complete.", flush=True)

            # --- Step 2: Initialize Cloud Clients, Embedding Model & Checkpointer ---
            print(f"--- {__name__}: STEP 2 - GCP Clients, Embedding, Checkpointer ---", flush=True)
            logger.info("Step 2: Initializing GCP Clients, Embedding Model, and Checkpointer...")
            # Check if essential libraries were loaded successfully earlier
            if not GCP_CLIENTS_LOADED or not DB_CONNECTOR_LOADED or not LANGGRAPH_LIBS_LOADED:
                raise ImportError("Required GCP, DB Connector, or LangGraph libraries failed to import earlier. Cannot proceed.")

            project_id = rix_config.get_config("GOOGLE_CLOUD_PROJECT")
            location = rix_config.get_config("GOOGLE_CLOUD_LOCATION")
            firestore_database_id = rix_config.get_config("FIRESTORE_DATABASE_ID", "(default)")
            print(f"--- {__name__}: Using GCP Project '{project_id}', Location '{location}'", flush=True)
            print(f"--- {__name__}: Using Firestore Database ID: '{firestore_database_id}'", flush=True)
            if not project_id or not location:
                raise ValueError("GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION missing in config.")

            # 2a: Firestore Client & LangGraph Checkpointer
            print(f"--- {__name__}: Initializing Firestore Client & Checkpointer (DB: '{firestore_database_id}')...", flush=True)
            fs_client = None
            try:
                # Instantiate Firestore client
                fs_client = firestore.Client(project=project_id, database=firestore_database_id)
                # Test connection (optional but recommended)
                print(f"--- {__name__}: Testing Firestore connection...", flush=True)
                # Simple check: list collections (limit 1 for efficiency)
                _ = list(fs_client.collections())
                logger.info(f"Firestore client initialized successfully for database '{firestore_database_id}'.")
                RixState.firestore_db_client = fs_client # Store client reference in global state
                print(f"--- {__name__}: Firestore Client OK.", flush=True)

                # Initialize LangGraph Firestore Checkpointer
                # It needs the Firestore client instance.
                RixState.firestore_checkpointer = FirestoreSaver(client=fs_client)
                logger.info("LangGraph Firestore checkpointer initialized successfully.")
                print(f"--- {__name__}: LangGraph Checkpointer OK.", flush=True)
            except Exception as fs_chk_e:
                logger.error(f"Failed to initialize Firestore client or Checkpointer: {fs_chk_e}", exc_info=True)
                print(f"--- {__name__}: EXCEPTION Firestore/Checkpointer Init: {fs_chk_e}", flush=True)
                raise ConnectionError("Firestore client or Checkpointer initialization failed.") from fs_chk_e

            # 2b: Vertex AI SDK Init & TEXT Embedding Model
            print(f"--- {__name__}: Initializing Vertex AI SDK & TEXT Embedding Model...", flush=True)
            try:
                logger.info("Initializing Vertex AI SDK...")
                # Initialize the Vertex AI SDK (required before using Vertex AI models)
                vertexai.init(project=project_id, location=location)
                # aiplatform.init(project=project_id, location=location) # Also works, vertexai.init is often preferred for generative models
                print(f"--- {__name__}: Vertex AI SDK initialized (vertexai.init).", flush=True)

                # Load the specific Text Embedding Model from config
                embed_model_name_from_config = rix_config.get_config("EMBEDDING_MODEL") # e.g., "text-embedding-005"
                if not embed_model_name_from_config:
                    raise ValueError("EMBEDDING_MODEL identifier not set in config.json.")
                
                final_embed_model_name = embed_model_name_from_config # Use the name directly from config
                logger.info(f"Attempting to load TEXT embedding model: '{final_embed_model_name}' using TextEmbeddingModel class.")
                print(f"--- {__name__}: Loading TEXT Embedding Model: {final_embed_model_name}...", flush=True)

                # Ensure the class was actually imported
                if TextEmbeddingModel is object or TextEmbeddingModel is None:
                     raise ImportError("TextEmbeddingModel class was not imported successfully.")

                # Instantiate the embedding model using from_pretrained
                embedding_model_instance = TextEmbeddingModel.from_pretrained(final_embed_model_name)
                RixState.vertex_embedding_model = embedding_model_instance # Store model instance in global state
                logger.info(f"Text Embedding model '{final_embed_model_name}' loaded into RixState.")
                print(f"--- {__name__}: Model instance created. Testing embedding...", flush=True)

                # Test the embedding model and CHECK DIMENSION
                test_embeddings = RixState.vertex_embedding_model.get_embeddings(["Rix test"])
                if not test_embeddings or not test_embeddings[0].values:
                     raise ValueError(f"Test query failed for text embedding model '{final_embed_model_name}'. No embeddings returned.")

                # --- *** Explicit Dimension Check *** ---
                actual_dimension = len(test_embeddings[0].values)
                logger.info(f"Loaded embedding model '{final_embed_model_name}'. Test embedding produced dimension: {actual_dimension}")
                if actual_dimension != EXPECTED_EMBEDDING_DIMENSION:
                    error_msg = (f"CRITICAL DIMENSION MISMATCH: Embedding model '{final_embed_model_name}' "
                                 f"produced dimension {actual_dimension}, but expected {EXPECTED_EMBEDDING_DIMENSION} "
                                 f"(based on code/DB schema). Update EXPECTED_EMBEDDING_DIMENSION in "
                                 f"initialization.py, pgvector_memory.py, and Cloud SQL schema.")
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                # --- *** End Dimension Check *** ---

                logger.info(f"Vertex AI SDK initialized and TEXT Embedding model '{final_embed_model_name}' OK (Dimension: {actual_dimension}).")
                print(f"--- {__name__}: Vertex AI TEXT Embedding OK (Dimension: {actual_dimension}).", flush=True)
            except Exception as vertex_e:
                 logger.error(f"Failed to initialize Vertex AI or load/test Embedding Model: {vertex_e}", exc_info=True)
                 print(f"--- {__name__}: EXCEPTION Vertex Init/Embedding Load: {vertex_e}", flush=True)
                 raise ConnectionError("Vertex AI / TEXT Embedding Model initialization failed.") from vertex_e

            # 2c: Cloud SQL Connection Pool Initialization (Synchronous Pool)
            # Note: Uses cloud-sql-python-connector with pg8000 for a standard SQLAlchemy sync engine.
            # If async DB operations are strictly needed later, this needs adjustment to create an async pool.
            print(f"--- {__name__}: Initializing Cloud SQL Connection Pool (Sync)...", flush=True)
            logger.info("Initializing Cloud SQL Connection Pool (Sync)...")
            db_user = rix_config.get_config("DB_USER")
            db_name = rix_config.get_config("DB_NAME")
            db_instance_connection_name = rix_config.get_config("DB_INSTANCE_CONNECTION_NAME")
            db_password_secret_name = rix_config.get_config("DB_PASSWORD_SECRET_NAME")
            if not all([db_user, db_name, db_instance_connection_name, db_password_secret_name]):
                raise ValueError("One or more Cloud SQL config keys (DB_USER, DB_NAME, DB_INSTANCE_CONNECTION_NAME, DB_PASSWORD_SECRET_NAME) missing.")

            db_password = None
            connector = Connector() # Instantiate the Cloud SQL connector
            
            # Fetch DB Password from Secret Manager
            try:
                print(f"--- {__name__}: Fetching DB secret: {db_password_secret_name}", flush=True)
                secret_client = SecretManagerServiceClient()
                response = secret_client.access_secret_version(name=db_password_secret_name)
                db_password = response.payload.data.decode("UTF-8")
                logger.info("DB password fetched successfully from Secret Manager.")
                print(f"--- {__name__}: DB Secret OK.", flush=True)
            except Exception as secret_e:
                logger.error(f"Failed to fetch DB password from Secret Manager '{db_password_secret_name}': {secret_e}", exc_info=True)
                print(f"--- {__name__}: EXCEPTION DB Secret: {secret_e}", flush=True)
                raise ConnectionError("Failed to get DB password from Secret Manager.") from secret_e
            if not db_password:
                raise ValueError("DB password fetched from Secret Manager is empty.")

            # Create Connection Pool using SQLAlchemy and Cloud SQL Connector
            try:
                print(f"--- {__name__}: Setting up Cloud SQL Connector function for instance '{db_instance_connection_name}'...", flush=True)
                # Define a function that returns a DBAPI connection using the connector
                def getconn() -> pg8000.dbapi.Connection: # Type hint for clarity
                    # Connect using the Cloud SQL connector; IPTypes.PUBLIC is common, adjust if using Private IP/PSC
                    conn: pg8000.dbapi.Connection = connector.connect(
                        db_instance_connection_name,
                        "pg8000", # Driver name
                        user=db_user,
                        password=db_password,
                        db=db_name,
                        ip_type=IPTypes.PUBLIC # Or IPTypes.PRIVATE if using VPC Native / Private IP
                    )
                    return conn

                print(f"--- {__name__}: Creating SQLAlchemy Engine (connection pool)...", flush=True)
                # Create a SQLAlchemy engine which manages a connection pool
                # The URL format "postgresql+pg8000://" tells SQLAlchemy to use pg8000 driver
                # The creator=getconn argument tells SQLAlchemy to use our connector function to get new connections
                pool = sqlalchemy.create_engine(
                    "postgresql+pg8000://",
                    creator=getconn,
                    pool_size=5,          # Max number of connections in the pool
                    max_overflow=2,       # How many extra connections allowed temporarily
                    pool_timeout=30,      # Seconds to wait for a connection before timing out
                    pool_recycle=1800     # Recycle connections after 30 minutes (helps prevent stale connections)
                )

                # Test the connection pool
                print(f"--- {__name__}: Testing Cloud SQL pool connection...", flush=True)
                connection = pool.connect() # Get a connection from the pool
                connection.close()          # Return the connection to the pool
                print(f"--- {__name__}: Cloud SQL Pool connection test OK.", flush=True)

                RixState.sql_engine_pool = pool # Store the pool instance in global state
                logger.info("Cloud SQL connection pool created and tested successfully.")
                print(f"--- {__name__}: Cloud SQL Pool OK.", flush=True)
            except Exception as db_conn_e:
                logger.error(f"Failed to create or test Cloud SQL connection pool: {db_conn_e}", exc_info=True)
                print(f"--- {__name__}: EXCEPTION DB Pool/Connect: {db_conn_e}", flush=True)
                # Attempt to close connector if pool creation failed after connector was instantiated
                if connector:
                    try: connector.close()
                    except Exception as close_e: print(f"Error closing connector after DB pool failure: {close_e}")
                raise ConnectionError("Cloud SQL pool creation or connection test failed.") from db_conn_e

            print(f"--- {__name__}: STEP 2 Complete.", flush=True)

            # --- Step 3: Load Soul Prompts (DEFERRED in V53.0) ---
            # In the hybrid model, souls will likely be loaded by the Cloud Run services themselves.
            # The orchestrator doesn't need direct access to the prompts globally.
            print(f"--- {__name__}: STEP 3 - Skip Loading Souls Globally ---", flush=True)
            logger.info("Step 3: Skipping global soul prompt loading (handled by agent services).")
            RixState.soul_manager_prompt = None
            RixState.soul_thinker_prompt = None
            RixState.soul_classifier_prompt = None
            RixState.soul_memory_writer_prompt = None
            print(f"--- {__name__}: STEP 3 Complete.", flush=True)

            # --- Step 4: Initialize Tool System ---
            # The local orchestrator might still need tool *schemas* even if execution is remote.
            # tool_manager.initialize() loads schemas from implementations/.
            print(f"--- {__name__}: STEP 4 - Initialize Tool System ---", flush=True)
            logger.info("Step 4: Initializing Tool System (loading schemas)...")
            if not Rix_tools_play: # Check if tool manager module was imported
                raise RuntimeError("Tool Manager (Rix_Brain/tools/tool_manager.py) not loaded or failed import.")
            try:
                tools_init_ok = Rix_tools_play.initialize() # Call initialize function in tool_manager
                # Optionally log number of tools/schemas loaded for verification
                num_tools = len(getattr(Rix_tools_play, 'FUNCTION_REGISTRY', {}))
                num_schemas = len(getattr(Rix_tools_play, 'SCHEMA_REGISTRY', {}))
                if tools_init_ok:
                    logger.info(f"Tool System initialized successfully ({num_tools} functions / {num_schemas} schemas registered).")
                    print(f"--- {__name__}: STEP 4 Complete.", flush=True)
                else:
                    logger.error("Tool System initialization failed (returned False). Check tool_manager logs.")
                    # Decide if this is a fatal error - maybe Rix can run without tools initially?
                    # For now, log error and continue.
                    print(f"--- {__name__}: WARNING - Tool System Init returned False.", flush=True)

            except Exception as tool_init_e:
                logger.error(f"Error during Tool System initialization: {tool_init_e}", exc_info=True)
                print(f"--- {__name__}: EXCEPTION Tool Init: {tool_init_e}", flush=True)
                # Decide if this is fatal
                # raise RuntimeError("Tool System initialization failed with exception.") from tool_init_e

            # --- Step 5: Final Success Check ---
            print(f"--- {__name__}: STEP 5 - Final Check & Success ---", flush=True)
            # Verify all essential components are in RixState
            if not all([RixState.firestore_db_client, RixState.firestore_checkpointer,
                        RixState.vertex_embedding_model, RixState.sql_engine_pool]):
                missing = [
                    k for k, v in {
                        "firestore_db_client": RixState.firestore_db_client,
                        "firestore_checkpointer": RixState.firestore_checkpointer,
                        "vertex_embedding_model": RixState.vertex_embedding_model,
                        "sql_engine_pool": RixState.sql_engine_pool
                    }.items() if v is None
                ]
                error_msg = f"Final check failed. Essential components missing in RixState: {missing}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # All checks passed, mark as initialized
            RixState.system_initialized = True
            RixState.initialization_error = None # Clear any "Initialization started" message
            logger.info(f"--- Rix Core Initialization Successful (V1.9.4 - Text Embedding {EXPECTED_EMBEDDING_DIMENSION}d) ---")
            print(f"--- {__name__}: SUCCESS - Rix Core Initialized.", flush=True)
            return True

        except Exception as e:
            # Catch-all for any unexpected error during the initialization steps
            init_fail_msg = f"Initialization Failed: {type(e).__name__} - {str(e)}"
            print(f"--- {__name__}: FATAL EXCEPTION in initialize_core: {init_fail_msg}", flush=True)
            logger.exception(f"--- Rix Core Initialization Failed --- : {init_fail_msg}")
            
            # Ensure state reflects failure
            if RixState: # Check if RixState module itself loaded
                 RixState.initialization_error = init_fail_msg
                 RixState.system_initialized = False
                 # Clear any partially initialized components to prevent inconsistent state
                 RixState.firestore_db_client = None
                 RixState.vertex_embedding_model = None
                 RixState.sql_engine_pool = None
                 RixState.firestore_checkpointer = None
            
            # Attempt to clean up Cloud SQL connector if it was created
            if connector:
                try:
                    connector.close()
                    logger.info("Cloud SQL connector closed after initialization failure.")
                except Exception as close_e:
                    logger.error(f"Error closing Cloud SQL connector after initialization failure: {close_e}")

            return False # Signal that initialization failed

# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\initialization.py ---