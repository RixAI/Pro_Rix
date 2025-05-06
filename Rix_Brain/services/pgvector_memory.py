# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\services\pgvector_memory.py ---
# Version: 1.2.0 (V52.2 - Multimodal Embedding Support) - Formerly rix_vector_pg.py
# Author: Vishal Sharma / Gemini
# Date: May 6, 2025
# Description: Handles interaction with Cloud SQL (PostgreSQL + pgvector) for
#              Rix vector memory, supporting Multimodal Embeddings (1408 dim).

print(f"--- Loading Module: {__name__} (V1.2.0 - Multimodal) ---", flush=True)

import logging
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional, Tuple

# --- Rix Core Module Imports (V52 Structure) ---
try:
    from Rix_Brain.core import global_state as RixState # Needed to get SQL Pool
    from Rix_Brain.core import utils as rix_utils # For timestamp
    RIX_BRAIN_LOADED = True
    print(f"--- {__name__}: Rix Brain core imports successful.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Import Fail: {e}", flush=True)
    RIX_BRAIN_LOADED = False
    RixState, rix_utils = None, None

# --- Third-party & GCP Imports ---
import sqlalchemy
try:
    from sqlalchemy.engine import Engine
    from sqlalchemy import text as sql_text
    # Import the specific embedding class for type hints and checks
    from vertexai.vision_models import MultiModalEmbeddingModel # <-- Import Multimodal model class
    DB_LIBS_AVAILABLE = True
    print(f"--- {__name__}: SQLAlchemy & Vertex AI MultiModalEmbeddingModel imported.", flush=True)
except ImportError as e:
    print(f"WARNING: {__name__} Failed import SQLAlchemy/Vertex AI MultiModalEmbeddingModel: {e}.", flush=True)
    Engine = object; sql_text = lambda x: x; MultiModalEmbeddingModel = object # Fallbacks
    DB_LIBS_AVAILABLE = False

# --- Logging Setup ---
logger = logging.getLogger("Rix_Heart")

# --- Constants ---
TABLE_NAME = "rix_memories"
EXPECTED_DIMENSION = 1408 # Dimension for multimodalembedding@001

# === Schema Definition Reminder ===
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE rix_memories (
#     memory_id UUID PRIMARY KEY,
#     embedding vector(1408),  -- *** MUST BE 1408 ***
#     memory_text TEXT,
#     memory_category VARCHAR(255),
#     metadata JSONB,
#     created_timestamp TIMESTAMPTZ DEFAULT now()
# );
# CREATE INDEX ON rix_memories USING hnsw (embedding vector_l2_ops); -- Or ivfflat
# =======================================

# --- Helper Function to get DB Pool ---
# NOTE: Embedder is now assumed to be passed directly to functions needing it,
# as different services might initialize it. Initialization module no longer stores it globally.
def _get_db_pool() -> Optional[Engine]:
    """Retrieves DB pool from RixState."""
    if not RIX_BRAIN_LOADED or not RixState:
        logger.error(f"{__name__}: RixState not available.")
        return None
    if not DB_LIBS_AVAILABLE:
        logger.error(f"{__name__}: Required DB libs not available.")
        return None

    # Use standard attribute name from global_state V1.1.0+
    db_pool = getattr(RixState, 'sql_engine_pool', None)

    if not db_pool:
        logger.error(f"{__name__}: DB Connection Pool missing in RixState.")
        return None
    if not isinstance(db_pool, Engine):
         logger.error(f"{__name__}: sql_engine_pool in RixState is not type Engine.")
         return None # Return None if invalid

    return db_pool

# --- Core Vector DB Functions ---

def add_to_vector_memory(
    embed_model_instance: MultiModalEmbeddingModel, # Pass the initialized model instance
    memory_text: str,
    metadata: Dict[str, Any],
    memory_category: str,
    memory_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
    """
    Adds or updates a memory entry using Multimodal embeddings.
    Requires an initialized MultiModalEmbeddingModel instance.
    """
    log_prefix = f"[{__name__}({memory_category})]"
    db_pool = _get_db_pool()

    # Validate inputs
    if not db_pool: logger.error(f"{log_prefix} DB Pool unavailable."); return False, None
    if not embed_model_instance or not isinstance(embed_model_instance, MultiModalEmbeddingModel):
        logger.error(f"{log_prefix} Invalid or missing MultiModalEmbeddingModel instance provided.")
        return False, None
    if not memory_text or not memory_category:
        logger.error(f"{log_prefix} Invalid input: memory_text and category required.")
        return False, None

    datapoint_id = memory_id or str(uuid.uuid4())
    metadata = metadata if isinstance(metadata, dict) else {}
    metadata['memory_id'] = datapoint_id
    if 'timestamp_utc' not in metadata:
        metadata['timestamp_utc'] = (rix_utils.get_current_timestamp() if rix_utils else
                                     datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z')

    # --- Get Multimodal Embedding ---
    vector = None
    try:
        logger.debug(f"{log_prefix} Embedding text ID: {datapoint_id} using Multimodal...")
        # Use the get_embeddings method for multimodal model
        response = embed_model_instance.get_embeddings(
            contextual_text=memory_text,
            dimension=EXPECTED_DIMENSION # Specify required dimension
            # Add image= or video= arguments here if embedding multimodal content
        )
        # Extract the text embedding vector
        if not response or not response.text_embedding:
            logger.error(f"{log_prefix} Multimodal embedding failed for ID: {datapoint_id}. No text_embedding found.")
            return False, datapoint_id
        vector = response.text_embedding
        if len(vector) != EXPECTED_DIMENSION:
             logger.error(f"{log_prefix} Embedding dimension mismatch! Expected {EXPECTED_DIMENSION}, got {len(vector)}. Check model/DB schema.")
             # Don't save if dimension is wrong
             return False, datapoint_id
        logger.debug(f"{log_prefix} Embedded ID: {datapoint_id} (Dim: {len(vector)}).")
    except Exception as e:
        logger.error(f"{log_prefix} Multimodal embedding error ID {datapoint_id}: {e}", exc_info=True)
        return False, datapoint_id # Return ID as it was generated

    # --- SQL Upsert (Ensure CAST uses correct dimension if schema requires it) ---
    # The CAST(.. AS vector) typically doesn't need dimension specified if column has it.
    sql = f"""
        INSERT INTO {TABLE_NAME} (memory_id, embedding, memory_text, memory_category, metadata, created_timestamp)
        VALUES (:mem_id, CAST(:embed AS vector), :mem_text, :mem_cat, CAST(:meta AS jsonb), now())
        ON CONFLICT (memory_id) DO UPDATE SET
            embedding = EXCLUDED.embedding, memory_text = EXCLUDED.memory_text,
            memory_category = EXCLUDED.memory_category, metadata = EXCLUDED.metadata,
            created_timestamp = now();
    """
    vector_str = None; metadata_json = None
    try:
        vector_str = '[' + ','.join(map(str, vector)) + ']' # Convert float list to string '[...]'
        metadata_json = json.dumps(metadata)
    except TypeError as json_e:
        logger.error(f"{log_prefix} Metadata serialization fail ID {datapoint_id}: {json_e}. Storing str.", exc_info=False)
        metadata_json = json.dumps({"unserializable_metadata": str(metadata)})
    except Exception as format_e:
        logger.error(f"{log_prefix} Vector/meta format fail ID {datapoint_id}: {format_e}", exc_info=True)
        return False, datapoint_id

    params_dict = {
        "mem_id": datapoint_id, "embed": vector_str, "mem_text": memory_text,
        "mem_cat": memory_category, "meta": metadata_json
    }

    # --- Execute Upsert ---
    conn = None; trans = None
    try:
        logger.info(f"{log_prefix} Upserting vector ID: {datapoint_id}...")
        conn = db_pool.connect()
        trans = conn.begin()
        conn.execute(sql_text(sql), params_dict)
        trans.commit()
        logger.info(f"{log_prefix} Cloud SQL Upsert OK ID: {datapoint_id}.")
        return True, datapoint_id
    # ... (Keep exact same error handling and finally block as V1.1.0) ...
    except sqlalchemy.exc.SQLAlchemyError as db_e:
        logger.error(f"{log_prefix} DB error Upsert ID {datapoint_id}: {db_e}", exc_info=True)
        if trans:
            try: trans.rollback(); logger.warning(f"{log_prefix} Transaction rolled back.")
            except Exception as rb_e: logger.error(f"{log_prefix} Rollback error: {rb_e}")
        return False, datapoint_id
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected Upsert error ID {datapoint_id}: {e}", exc_info=True)
        if trans:
             try: trans.rollback()
             except Exception as rb_e: logger.error(f"{log_prefix} Rollback error: {rb_e}")
        return False, datapoint_id
    finally:
        if conn: conn.close()


def recall_memory(
    embed_model_instance: MultiModalEmbeddingModel, # Pass the initialized model instance
    query_text: str,
    top_n: int = 5,
    filter_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
    """
    Recalls memories using Multimodal embeddings, optionally filtering.
    Requires an initialized MultiModalEmbeddingModel instance.
    """
    log_prefix = f"[{__name__}({filter_category or 'All'})]"
    recalled_memories = []
    db_pool = _get_db_pool()

    # Validate inputs
    if not db_pool: logger.error(f"{log_prefix} DB Pool unavailable."); return []
    if not embed_model_instance or not isinstance(embed_model_instance, MultiModalEmbeddingModel):
        logger.error(f"{log_prefix} Invalid or missing MultiModalEmbeddingModel instance provided.")
        return []
    if not query_text: logger.error(f"{log_prefix} Query text required."); return []
    if not isinstance(top_n, int) or top_n <= 0: logger.warning(f"{log_prefix} Invalid top_n ({top_n}). Using 5."); top_n = 5

    # --- Get Multimodal Embedding for Query ---
    query_vector = None
    try:
        logger.debug(f"{log_prefix} Embedding query text using Multimodal...")
        response = embed_model_instance.get_embeddings(
            contextual_text=query_text,
            dimension=EXPECTED_DIMENSION # Specify required dimension
        )
        if not response or not response.text_embedding:
            logger.error(f"{log_prefix} Query embedding failed. No text_embedding found.")
            return []
        query_vector = response.text_embedding
        if len(query_vector) != EXPECTED_DIMENSION:
             logger.error(f"{log_prefix} Query embedding dimension mismatch! Expected {EXPECTED_DIMENSION}, got {len(query_vector)}.")
             return []
        logger.debug(f"{log_prefix} Query embedded (Dim: {len(query_vector)}).")
    except Exception as e:
        logger.error(f"{log_prefix} Query multimodal embedding error: {e}", exc_info=True)
        return []

    # --- SQL Query ---
    distance_operator = "<->" # L2 distance for pgvector
    sql_select = f"SELECT memory_id, memory_text, metadata, embedding {distance_operator} CAST(:query_vec_str AS vector({EXPECTED_DIMENSION})) AS distance" # Explicit dimension in CAST can help
    sql_from = f"FROM {TABLE_NAME}"
    sql_where_parts = []
    sql_order_by = f"ORDER BY distance ASC"
    sql_limit = "LIMIT :limit_num"

    query_vector_str = None
    try: query_vector_str = '[' + ','.join(map(str, query_vector)) + ']'
    except Exception as format_e: logger.error(f"{log_prefix} Query vector format fail: {format_e}", exc_info=True); return []

    params_dict = {"query_vec_str": query_vector_str, "limit_num": top_n}
    if filter_category:
        sql_where_parts.append("memory_category = :filter_cat")
        params_dict["filter_cat"] = filter_category

    sql_where = "WHERE " + " AND ".join(sql_where_parts) if sql_where_parts else ""
    final_sql = f"{sql_select} {sql_from} {sql_where} {sql_order_by} {sql_limit};"

    log_params = {k: (v[:50] + '...' if isinstance(v, str) and k == 'query_vec_str' else v) for k, v in params_dict.items()}
    logger.debug(f"{log_prefix} Exec SQL: {final_sql} | Params: {log_params}")

    # --- Execute Query ---
    conn = None
    try:
        logger.info(f"{log_prefix} Querying Cloud SQL (TopN: {top_n})...")
        conn = db_pool.connect()
        result_proxy = conn.execute(sql_text(final_sql), params_dict)
        results = result_proxy.fetchall()
        result_proxy.close()
        logger.info(f"{log_prefix} Found {len(results)} neighbors.")

        for row in results:
            # ... (Keep metadata parsing logic exactly as it was in V1.1.0) ...
            mem_id, mem_text, mem_metadata_raw, distance = row[0], row[1], row[2], row[3]
            mem_metadata = None
            if isinstance(mem_metadata_raw, str):
                try: mem_metadata = json.loads(mem_metadata_raw)
                except json.JSONDecodeError: logger.warning(f"{log_prefix} Decode metadata fail ID: {mem_id}."); mem_metadata = {"raw_metadata": mem_metadata_raw}
            elif isinstance(mem_metadata_raw, dict): mem_metadata = mem_metadata_raw
            else: logger.warning(f"{log_prefix} Unexpected metadata type ID: {mem_id}."); mem_metadata = {"raw_metadata": str(mem_metadata_raw)}
            recalled_memories.append({ "id": mem_id, "distance": float(distance), "text": mem_text, "metadata": mem_metadata })
            logger.debug(f"{log_prefix} Recalled ID: {mem_id}, Distance: {distance:.4f}")

    # ... (Keep exact same error handling and finally block as V1.1.0) ...
    except sqlalchemy.exc.SQLAlchemyError as db_e: logger.error(f"{log_prefix} DB error Query: {db_e}", exc_info=True)
    except Exception as e: logger.error(f"{log_prefix} Unexpected error Query: {e}", exc_info=True)
    finally:
        if conn: conn.close()

    return recalled_memories


print(f"--- Module Defined: {__name__} (V1.2.0 - Multimodal) ---", flush=True)

# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\services\pgvector_memory.py ---