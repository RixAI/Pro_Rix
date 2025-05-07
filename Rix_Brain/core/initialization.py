# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\initialization.py ---
# Version: 1.9.9 (V53.0 - Explicit SA Key, Corrected FS Test, FS Checkpointer Attempt)

print(f"--- Loading Module: {__name__} (V1.9.9 - Explicit SA Key, Corrected FS Test) ---", flush=True)

import logging
import json
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# --- Rix Core Module Imports ---
try:
    from . import global_state as RixState
    from . import config_manager as rix_config
    from . import utils as rix_utils
    from Rix_Brain.tools import tool_manager as Rix_tools_play
    RIX_BRAIN_LOADED = True
    print(f"--- {__name__}: Rix Brain core imports successful.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Core Import Fail: {e}", flush=True)
    RIX_BRAIN_LOADED = False
    raise

# --- Required Libs ---
FIRESTORE_CLIENT_LIB_LOADED = False
LANGCHAIN_FIRESTORE_SAVER_LOADED = False  # To track if the specific checkpointer lib is available
GCP_CLIENTS_LOADED = False
DB_CONNECTOR_LOADED = False

firestore = None
FirestoreSaver = None  # Initialize to None
aiplatform = None
vertexai = None
TextEmbeddingModel = None
SecretManagerServiceClient = None
sqlalchemy = None
pg8000 = None
Connector = None
IPTypes = None
service_account_creds_module = None  # For google.oauth2.service_account

print(f"--- {__name__}: Importing GCP, DB, & Potentially LangChain Firestore Libs...", flush=True)
try:
    from google.cloud import firestore
    FIRESTORE_CLIENT_LIB_LOADED = True
    logger = logging.getLogger("Rix_Heart")  # Initialize logger early
    logger.info("Firestore client library (google.cloud.firestore) imported.")

    try:
        from langchain_google_firestore import FirestoreSaver  # Correct import path
        LANGCHAIN_FIRESTORE_SAVER_LOADED = True
        logger.info("LangChain FirestoreSaver (langchain_google_firestore) imported.")
    except ImportError:
        logger.warning("langchain_google_firestore.FirestoreSaver not found. Firestore checkpointer will not be available. Graph will likely use MemorySaver.")

    import google.cloud.aiplatform as aiplatform
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    from google.cloud.secretmanager import SecretManagerServiceClient
    from google.oauth2 import service_account  # For explicit SA key loading
    service_account_creds_module = service_account  # Assign for use
    GCP_CLIENTS_LOADED = True

    from google.cloud.sql.connector import Connector, IPTypes
    import sqlalchemy
    import pg8000
    DB_CONNECTOR_LOADED = True

    logger.info("Initialization Module: Required GCP & DB libraries imported successfully.")
    print(f"--- {__name__}: Lib Imports OK.", flush=True)
except ImportError as e:
    print(f"--- {__name__}: WARNING - Lib Import Fail: {e}. Some initializations might fail.", flush=True)
    if 'logger' not in locals():
        logger = logging.getLogger("Rix_Heart_Fallback")
    logger.warning(f"Init WARNING: Failed import of some libraries: {e}.", exc_info=True)

EXPECTED_EMBEDDING_DIMENSION = 768
logger = logging.getLogger("Rix_Heart")
_init_lock = threading.Lock()

def _get_service_account_key_path() -> Optional[str]:
    """
    Determines the path to the service account key JSON file.
    1. Tries rix_config.get_google_app_credentials().
    2. Falls back to a hardcoded path if the first fails.
    Returns the path as a string if found and valid, else None.
    """
    path_from_config = None
    if rix_config and hasattr(rix_config, 'get_google_app_credentials'):
        path_from_config = rix_config.get_google_app_credentials()

    if path_from_config and os.path.exists(path_from_config):
        logger.info(f"Using SA Key Path from config_manager: {path_from_config}")
        return path_from_config
    else:
        if path_from_config:
            logger.warning(f"SA Key Path from config_manager ('{path_from_config}') not found or invalid.")
        try:
            current_file_path = Path(__file__).resolve()
            rix_brain_dir_path = current_file_path.parent.parent
            # *** Ensure this fallback filename is your NEW, ACTIVE key ***
            hardcoded_key_path = rix_brain_dir_path / "Rix_Auth" / "rixagi-c44b931dd345.json"  # YOUR NEW KEY

            if hardcoded_key_path.exists():
                logger.warning(f"Falling back to hardcoded SA Key Path: {str(hardcoded_key_path)}")
                return str(hardcoded_key_path)
            else:
                logger.error(f"Hardcoded SA Key Path does NOT exist: {str(hardcoded_key_path)}")
                return None
        except Exception as e_path:
            logger.error(f"Error constructing fallback SA Key Path: {e_path}")
            return None

def initialize_core() -> bool:
    print(f"--- {__name__}: ENTERING initialize_core (V1.9.9 - Explicit SA Key, Correct FS Test) ---", flush=True)
    with _init_lock:
        if getattr(RixState, 'system_initialized', False):
            logger.warning("Rix Core already initialized.")
            return True

        logger.info(f"--- Rix Core Initialization Start (V1.9.9) ---")
        RixState.system_initialized = False
        RixState.initialization_error = "Init started."
        RixState.firestore_db_client = None
        RixState.vertex_embedding_model = None
        RixState.sql_engine_pool = None
        RixState.firestore_checkpointer = None

        connector: Optional[Connector] = None
        explicit_credentials: Optional[service_account_creds_module.Credentials] = None

        rix_config.ensure_loaded()
        sa_key_file_path = _get_service_account_key_path()

        if sa_key_file_path and service_account_creds_module:  # Check if service_account module loaded
            try:
                explicit_credentials = service_account_creds_module.Credentials.from_service_account_file(sa_key_file_path)
                logger.info(f"Successfully loaded explicit credentials from: {sa_key_file_path}")
            except Exception as e_sa_load:
                logger.error(f"Failed to load explicit credentials from {sa_key_file_path}: {e_sa_load}. Will attempt ADC.", exc_info=True)
                explicit_credentials = None
        else:
            if not service_account_creds_module:
                logger.warning("google.oauth2.service_account module not loaded, cannot use explicit SA key.")
            logger.warning("Service Account Key Path not found or SA module missing. Will attempt Application Default Credentials.")

        try:
            # Step 1: Config & Logging
            print(f"--- {__name__}: STEP 1 - Config & Logging ---", flush=True)
            logger.info("Step 1: Config & Logging...")
            log_level_str = rix_config.get_config("LOG_LEVEL", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            logging.getLogger("Rix_Heart").setLevel(log_level)
            logging.getLogger("Rix_Brain").setLevel(log_level)
            try:
                logging.getLogger("RixCli").setLevel(log_level)
            except Exception:
                pass
            logger.info(f"Logging level set to: {log_level_str}")
            print(f"--- {__name__}: STEP 1 Complete.", flush=True)

            # Step 2: GCP Clients, Embedding, Checkpointer
            print(f"--- {__name__}: STEP 2 - GCP Clients, Embedding, Checkpointer ---", flush=True)
            project_id = rix_config.get_config("GOOGLE_CLOUD_PROJECT")
            location = rix_config.get_config("GOOGLE_CLOUD_LOCATION")
            if not project_id or not location:
                raise ValueError("GCP Project/Location missing.")

            # 2a: Firestore Client & LangGraph Checkpointer
            if FIRESTORE_CLIENT_LIB_LOADED and firestore:
                firestore_database_id = rix_config.get_config("FIRESTORE_DATABASE_ID", "(default)")
                print(f"--- {__name__}: Initializing Firestore Client (DB: '{firestore_database_id}')...", flush=True)
                fs_client = None  # Declare here for broader scope
                try:
                    if explicit_credentials:
                        logger.info("Using EXPLICIT credentials for Firestore Client.")
                        fs_client = firestore.Client(project=project_id, credentials=explicit_credentials, database=firestore_database_id)
                    else:
                        logger.info("Using ADC for Firestore Client.")
                        fs_client = firestore.Client(project=project_id, database=firestore_database_id)

                    # Corrected Firestore client test
                    if not (fs_client and hasattr(fs_client, 'project') and fs_client.project == project_id):
                        raise ConnectionError(f"Firestore client object is None or project mismatch (expected {project_id}, got {getattr(fs_client, 'project', 'Attribute not found or client is None')}).")
                    logger.info(f"Firestore client successfully instantiated for project '{fs_client.project}'.")
                    print(f"--- {__name__}: Firestore Client OK.", flush=True)
                    RixState.firestore_db_client = fs_client

                    if LANGCHAIN_FIRESTORE_SAVER_LOADED and FirestoreSaver and fs_client:  # Check fs_client again
                        try:
                            RixState.firestore_checkpointer = FirestoreSaver(client=fs_client)
                            logger.info("LangGraph Firestore checkpointer initialized successfully.")
                            print(f"--- {__name__}: LangGraph Checkpointer OK.", flush=True)
                        except Exception as e_fss_create:
                            logger.error(f"Error creating FirestoreSaver checkpointer: {e_fss_create}. Checkpointer will be None.", exc_info=True)
                            RixState.firestore_checkpointer = None
                    else:
                        if not LANGCHAIN_FIRESTORE_SAVER_LOADED:
                            logger.warning("FirestoreSaver library not loaded.")
                        if not fs_client:
                            logger.warning("Firestore client (fs_client) is None.")
                        logger.warning("Firestore checkpointer will be None. Graph may use MemorySaver.")
                        RixState.firestore_checkpointer = None
                except Exception as fs_e:
                    logger.error(f"Failed to initialize Firestore client: {fs_e}", exc_info=True)
                    print(f"--- {__name__}: EXCEPTION Firestore Client Init: {fs_e}", flush=True)  # Changed from WARNING
                    RixState.firestore_db_client = None
                    RixState.firestore_checkpointer = None
                    # If Firestore is critical, you might re-raise here.
                    # For now, allow graph_pipeline to fallback to MemorySaver if checkpointer is None.
            else:
                logger.warning("Firestore client library (google.cloud.firestore) not loaded. Firestore features unavailable.")
                print(f"--- {__name__}: SKIPPING Firestore Client/Checkpointer Init (library not loaded).", flush=True)
                RixState.firestore_checkpointer = None

            # 2b: Vertex AI SDK Init & TEXT Embedding Model
            print(f"--- {__name__}: Initializing Vertex AI SDK & TEXT Embedding Model...", flush=True)
            try:
                if not GCP_CLIENTS_LOADED or not vertexai or not TextEmbeddingModel:
                    raise ImportError("Vertex AI libs not loaded.")
                if explicit_credentials:
                    logger.info("Using EXPLICIT credentials for Vertex AI SDK init.")
                    vertexai.init(project=project_id, location=location, credentials=explicit_credentials)
                else:
                    logger.info("Using ADC for Vertex AI SDK init.")
                    vertexai.init(project=project_id, location=location)
                print(f"--- {__name__}: Vertex AI SDK initialized.", flush=True)

                embed_model_name_from_config = rix_config.get_config("EMBEDDING_MODEL")
                if not embed_model_name_from_config:
                    raise ValueError("EMBEDDING_MODEL identifier not set.")
                RixState.vertex_embedding_model = TextEmbeddingModel.from_pretrained(embed_model_name_from_config)
                test_embeddings = RixState.vertex_embedding_model.get_embeddings(["Rix test"])
                if not test_embeddings or not test_embeddings[0].values:
                    raise ValueError("Test query failed for embedding model.")
                actual_dimension = len(test_embeddings[0].values)
                logger.info(f"Loaded embedding model '{embed_model_name_from_config}'. Test dimension: {actual_dimension}")
                if actual_dimension != EXPECTED_EMBEDDING_DIMENSION:
                    raise ValueError(f"Dimension Mismatch: {actual_dimension}D vs expected {EXPECTED_EMBEDDING_DIMENSION}D.")
                logger.info(f"Vertex AI SDK & Embedding model OK (Dimension: {actual_dimension}).")
                print(f"--- {__name__}: Vertex AI TEXT Embedding OK (Dimension: {actual_dimension}).", flush=True)
            except Exception as vertex_e:
                logger.error(f"Failed Vertex AI/Embedding init: {vertex_e}", exc_info=True)
                raise ConnectionError("Vertex AI / Embedding Model init failed.") from vertex_e

            # 2c: Cloud SQL Connection Pool
            print(f"--- {__name__}: Initializing Cloud SQL Connection Pool...", flush=True)
            db_user = rix_config.get_config("DB_USER")
            db_name = rix_config.get_config("DB_NAME")
            db_instance_connection_name = rix_config.get_config("DB_INSTANCE_CONNECTION_NAME")
            db_password_secret_name = rix_config.get_config("DB_PASSWORD_SECRET_NAME")
            if not all([db_user, db_name, db_instance_connection_name, db_password_secret_name]):
                raise ValueError("Missing Cloud SQL config keys.")
            db_password = None
            connector = Connector()
            try:
                if not GCP_CLIENTS_LOADED or not SecretManagerServiceClient:
                    raise ImportError("SecretManagerServiceClient not loaded.")
                print(f"--- {__name__}: Fetching DB secret: {db_password_secret_name}...", flush=True)
                if explicit_credentials:
                    logger.info("Using EXPLICIT credentials for Secret Manager Client.")
                    secret_client = SecretManagerServiceClient(credentials=explicit_credentials)
                else:
                    logger.info("Using ADC for Secret Manager Client.")
                    secret_client = SecretManagerServiceClient()
                response = secret_client.access_secret_version(name=db_password_secret_name)
                db_password = response.payload.data.decode("UTF-8")
                logger.info("DB password fetched.")
            except Exception as secret_e:
                logger.error(f"Failed Secret Manager: {secret_e}", exc_info=True)
                raise ConnectionError("Failed to get DB password from Secret Manager.") from secret_e
            if not db_password:
                raise ValueError("DB password empty.")
            try:
                if not DB_CONNECTOR_LOADED or not sqlalchemy or not pg8000:
                    raise ImportError("DB connector libs not loaded.")

                def getconn() -> pg8000.dbapi.Connection:
                    conn_obj: pg8000.dbapi.Connection = connector.connect(
                        db_instance_connection_name,
                        "pg8000",
                        user=db_user,
                        password=db_password,
                        db=db_name,
                        ip_type=IPTypes.PUBLIC
                    )
                    return conn_obj

                pool = sqlalchemy.create_engine(
                    "postgresql+pg8000://",
                    creator=getconn,
                    pool_size=5,
                    max_overflow=2,
                    pool_timeout=30,
                    pool_recycle=1800
                )
                connection = pool.connect()
                connection.close()
                RixState.sql_engine_pool = pool
                logger.info("Cloud SQL pool created successfully.")
            except Exception as db_conn_e:
                logger.error(f"Failed SQL Pool: {db_conn_e}", exc_info=True)
                if connector:
                    try:
                        connector.close()
                    except Exception:
                        pass
                raise ConnectionError("Cloud SQL pool failed.") from db_conn_e
            print(f"--- {__name__}: STEP 2 Complete.", flush=True)

            # Step 3 & 4 (Souls, Tools)
            print(f"--- {__name__}: STEP 3 - Skip Souls ---", flush=True)
            logger.info("Step 3: Skipping global soul loading.")
            print(f"--- {__name__}: STEP 3 Complete.", flush=True)
            print(f"--- {__name__}: STEP 4 - Init Tools ---", flush=True)
            logger.info("Step 4: Initializing Tool System...")
            if not Rix_tools_play:
                raise RuntimeError("Tool Manager module not loaded.")
            try:
                tools_init_ok = Rix_tools_play.initialize()
                if tools_init_ok:
                    logger.info(f"Tool System initialized.")
                else:
                    logger.error("Tool System init returned False.")
            except Exception as tool_init_e:
                logger.error(f"Error Tool Init: {tool_init_e}", exc_info=True)
            print(f"--- {__name__}: STEP 4 Complete.", flush=True)

            # Step 5: Final Success Check
            print(f"--- {__name__}: STEP 5 - Final Check ---", flush=True)
            if not all([RixState.vertex_embedding_model, RixState.sql_engine_pool]):  # Checkpointer might be None
                missing = [k for k, v in {
                    "vertex_embedding_model": RixState.vertex_embedding_model,
                    "sql_engine_pool": RixState.sql_engine_pool
                }.items() if v is None]
                raise RuntimeError(f"Final check failed. Missing RixState components: {missing}")
            RixState.system_initialized = True
            RixState.initialization_error = None
            logger.info(f"--- Rix Core Initialization Successful (V1.9.9) ---")
            print(f"--- {__name__}: SUCCESS - Rix Core Initialized.", flush=True)
            return True

        except Exception as e:
            init_fail_msg = f"Initialization Failed: {type(e).__name__} - {str(e)}"
            print(f"--- {__name__}: FATAL EXCEPTION in initialize_core: {init_fail_msg}", flush=True)
            logger.exception(f"--- Rix Core Initialization Failed --- : {init_fail_msg}")
            if RixState:
                RixState.initialization_error = init_fail_msg
                RixState.system_initialized = False
            if connector:
                try:
                    connector.close()
                except Exception as ce:
                    logger.error(f"Connector close error: {ce}")
            return False

# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\core\initialization.py ---