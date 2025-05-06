# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\orchestration\graph_pipeline.py ---
# Version: 1.1.5 (V53.0 - MEMORY SAVER + Enhanced Getter Logging)

print(f"--- Loading Module: {__name__} (V1.1.5 - MEMORY SAVER + Getter Logging) ---", flush=True)

import logging
from typing import TypedDict, List, Optional, Dict, Any, Literal, Tuple, Sequence

LANGGRAPH_FULLY_LOADED = False
StateGraph = None
END = None
MemorySaver = None

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_FULLY_LOADED = True
    print(f"--- {__name__}: LangGraph core AND MemorySaver imported successfully.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Failed to import LangGraph/MemorySaver: {e}. Ensure 'langgraph' is installed.", flush=True)
    class StateGraph_dummy:
        pass
    StateGraph = StateGraph_dummy
    class END_dummy:
        pass
    END = END_dummy
    class MemorySaver_dummy:
        pass
    MemorySaver = MemorySaver_dummy
    if 'TypedDict' not in locals():
        TypedDict = dict

try:
    from Rix_Brain.agents import nodes as agent_nodes
    print(f"--- {__name__}: Agent Nodes imported.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Failed to import Agent Nodes: {e}.", flush=True)
    agent_nodes = None

logger = logging.getLogger("Rix_Brain.orchestration")

class PipelineState(TypedDict, total=False):
    session_id: str
    user_input: Optional[str]
    classification: Optional[Literal["CHAT", "ASK", "WORK", "RCD"]]
    dispatch_instruction: Optional[str]
    current_tool_request: Optional[Dict]
    current_tool_result: Optional[Dict]
    agent_history: List[Tuple[str, str]]
    intermediate_steps: List[Any]
    final_response: Optional[str]
    error_message: Optional[str]
    rcd_plan: Optional[List[Dict]]
    rcd_current_step_index: int
    rcd_step_outputs: Dict[int, Any]
    next_node_target: Optional[str]


def create_rix_graph() -> Optional[StateGraph]:
    logger.info("create_rix_graph: Attempting to create graph...")  # ADDED LOG
    if not LANGGRAPH_FULLY_LOADED:
        logger.error("create_rix_graph: LangGraph/MemorySaver not loaded.")
        return None
    if not agent_nodes:
        logger.error("create_rix_graph: Agent nodes not loaded.")
        return None
    if MemorySaver is None or END is None or StateGraph is None:
        logger.error("create_rix_graph: Core LangGraph components failed to import properly.")
        return None

    checkpointer = MemorySaver()
    logger.info(f"create_rix_graph: Using MemorySaver instance {checkpointer} for checkpointing.")

    graph_builder = StateGraph(PipelineState)
    # ... (Add Nodes - Keep as before) ...
    try:
        graph_builder.add_node("classifier", agent_nodes.classifier_node)
        graph_builder.add_node("thinker", agent_nodes.thinker_node)
        graph_builder.add_node("tool_executor", agent_nodes.tool_executor_node)
        graph_builder.add_node("memory_writer", agent_nodes.memory_writer_node)
        graph_builder.add_node("finalizer", agent_nodes.finalizer_node)
        logger.info("create_rix_graph: All agent nodes added.")
    except Exception as e:
        logger.error(f"create_rix_graph: Error adding nodes: {e}", exc_info=True)
        return None

    # --- Define Edges (Keep as before) ---
    graph_builder.set_entry_point("classifier")

    def route_after_classifier(state: PipelineState) -> str:
        log_prefix = f"[{state.get('session_id')}] RouteAfterClassifier"
        classification = state.get("classification")
        error_message = state.get("error_message")
        logger.debug(f"{log_prefix}: Classification='{classification}', Error='{error_message}'")
        if error_message:
            return "finalizer"
        if classification in ["ASK", "WORK", "RCD"]:
            return "thinker"
        elif classification == "CHAT":
            return "memory_writer"
        else:
            return "finalizer"

    graph_builder.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"thinker": "thinker", "memory_writer": "memory_writer", "finalizer": "finalizer"}
    )

    def route_after_thinker(state: PipelineState) -> str:
        log_prefix = f"[{state.get('session_id')}] RouteAfterThinker"
        tool_request = state.get("current_tool_request")
        error_message = state.get("error_message")
        final_response_by_thinker = state.get("final_response")
        logger.debug(f"{log_prefix}: ToolRequest='{bool(tool_request)}', Error='{error_message}', DirectResponse='{bool(final_response_by_thinker)}'")
        if error_message:
            return "finalizer"
        if tool_request:
            return "tool_executor"
        if final_response_by_thinker:
            return "memory_writer"
        return "memory_writer"

    graph_builder.add_conditional_edges(
        "thinker",
        route_after_thinker,
        {"tool_executor": "tool_executor", "memory_writer": "memory_writer", "finalizer": "finalizer"}
    )

    def route_after_tool_executor(state: PipelineState) -> str:
        log_prefix = f"[{state.get('session_id')}] RouteAfterToolExecutor"
        tool_result = state.get("current_tool_result")
        error_message = state.get("error_message")
        tool_status = tool_result.get("status", "unknown") if isinstance(tool_result, dict) else "unknown"
        logger.debug(f"{log_prefix}: ToolStatus='{tool_status}', NodeError='{error_message}'")
        if error_message:
            return "finalizer"
        if tool_status == "failed":
            return "thinker"
        else:
            return "thinker"

    graph_builder.add_conditional_edges(
        "tool_executor",
        route_after_tool_executor,
        {"thinker": "thinker", "finalizer": "finalizer"}
    )

    graph_builder.add_edge("memory_writer", "finalizer")
    graph_builder.add_edge("finalizer", END)

    try:
        compiled_graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info(f"create_rix_graph: Rix LangGraph compiled successfully using MemorySaver. Type: {type(compiled_graph)}")  # ADDED LOG
        return compiled_graph
    except Exception as compile_e:
        logger.exception(f"create_rix_graph: FATAL: Failed to compile Rix LangGraph: {compile_e}")
        return None

# Module-level variable for the compiled graph
rix_graph_app: Optional[StateGraph] = None

def get_compiled_graph():
    global rix_graph_app
    logger.info("get_compiled_graph: Called.")  # ADDED LOG
    if rix_graph_app is None:
        logger.info("get_compiled_graph: rix_graph_app is None, attempting to create...")  # ADDED LOG
        if LANGGRAPH_FULLY_LOADED and agent_nodes:
            rix_graph_app = create_rix_graph()
            if rix_graph_app:
                logger.info(f"get_compiled_graph: create_rix_graph SUCCEEDED. rix_graph_app type: {type(rix_graph_app)}")  # ADDED LOG
            else:
                logger.error("get_compiled_graph: create_rix_graph FAILED and returned None.")  # ADDED LOG
        else:
            if not LANGGRAPH_FULLY_LOADED:
                logger.error("get_compiled_graph: Cannot create graph, LANGGRAPH_FULLY_LOADED is False.")
            if not agent_nodes:
                logger.error("get_compiled_graph: Cannot create graph, agent_nodes is None.")
    else:
        logger.info("get_compiled_graph: rix_graph_app already exists.")  # ADDED LOG
    return rix_graph_app

print(f"--- Module Defined: {__name__} (V1.1.5 - MEMORY SAVER + Getter Logging) ---", flush=True)
# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\orchestration\graph_pipeline.py ---