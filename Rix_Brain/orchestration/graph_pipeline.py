# --- START OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\orchestration\graph_pipeline.py ---
# Version: 1.1.6 (V57.0 - Uses manager_node as final step, MemorySaver)

print(f"--- Loading Module: {__name__} (V1.1.6 - manager_node, MemorySaver) ---", flush=True)

import logging
from typing import TypedDict, List, Optional, Dict, Any, Literal, Tuple, Sequence

LANGGRAPH_FULLY_LOADED = False
StateGraph, END, MemorySaver = None, None, None

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_FULLY_LOADED = True
    print(f"--- {__name__}: LangGraph core AND MemorySaver imported successfully.", flush=True)
except ImportError as e:
    print(f"FATAL: {__name__} Failed to import LangGraph/MemorySaver: {e}. Ensure 'langgraph' is installed.", flush=True)
    class StateGraph_dummy: pass; StateGraph = StateGraph_dummy # type: ignore
    class END_dummy: pass; END = END_dummy # type: ignore
    class MemorySaver_dummy: pass; MemorySaver = MemorySaver_dummy # type: ignore
    if 'TypedDict' not in locals(): TypedDict = dict # type: ignore

try:
    from Rix_Brain.agents import nodes as agent_nodes
    print(f"--- {__name__}: Agent Nodes imported.", flush=True)
except ImportError as e:
     print(f"FATAL: {__name__} Failed to import Agent Nodes: {e}.", flush=True)
     agent_nodes = None

logger = logging.getLogger("Rix_Brain.orchestration")

class PipelineState(TypedDict, total=False):
    session_id: str; user_input: Optional[str]; classification: Optional[Literal["CHAT", "ASK", "WORK", "RCD", "UNKNOWN"]]; dispatch_instruction: Optional[str]; current_tool_request: Optional[Dict]; current_tool_result: Optional[Dict]; agent_history: List[Tuple[str, str]]; intermediate_steps: List[Any]; final_response: Optional[str]; error_message: Optional[str]; rcd_plan: Optional[List[Dict]]; rcd_current_step_index: int; rcd_step_outputs: Dict[int, Any]; next_node_target: Optional[str]

def create_rix_graph() -> Optional[StateGraph]:
    logger.info("create_rix_graph: Attempting to create graph...")
    if not LANGGRAPH_FULLY_LOADED or not agent_nodes or MemorySaver is None or END is None or StateGraph is None:
        logger.error("create_rix_graph: Core LangGraph components or agent_nodes failed to load properly."); return None

    checkpointer = MemorySaver(); logger.info(f"create_rix_graph: Using MemorySaver instance {checkpointer}.")
    graph_builder = StateGraph(PipelineState)
    try:
        graph_builder.add_node("classifier", agent_nodes.classifier_node)
        graph_builder.add_node("thinker", agent_nodes.thinker_node)
        graph_builder.add_node("tool_executor", agent_nodes.tool_executor_node)
        graph_builder.add_node("memory_writer", agent_nodes.memory_writer_node)
        graph_builder.add_node("manager", agent_nodes.manager_node) # <-- USE "manager" node
        logger.info("create_rix_graph: All agent nodes added.")
    except Exception as e: logger.error(f"create_rix_graph: Error adding nodes: {e}", exc_info=True); return None

    graph_builder.set_entry_point("classifier")
    def route_after_classifier(state: PipelineState) -> str:
        # ... (same routing logic as V1.1.5, but error routes to "manager")
        classification = state.get("classification"); error_message = state.get("error_message")
        if error_message: return "manager" # Error goes to manager to formulate user error message
        if classification in ["ASK", "WORK", "RCD"]: return "thinker"
        elif classification == "CHAT": return "memory_writer" # CHAT goes to memory then manager
        return "manager" # Default/Unknown to manager

    graph_builder.add_conditional_edges("classifier", route_after_classifier, {
        "thinker": "thinker", "memory_writer": "memory_writer", "manager": "manager"
    })

    def route_after_thinker(state: PipelineState) -> str:
        # ... (same routing logic as V1.1.5, but error or completion routes to "memory_writer" then "manager")
        tool_request = state.get("current_tool_request"); error_message = state.get("error_message"); final_response_by_thinker = state.get("final_response")
        if error_message: return "manager" # Error from thinker goes to manager
        if tool_request: return "tool_executor"
        return "memory_writer" # Success (direct answer or no tool) goes to memory writer

    graph_builder.add_conditional_edges("thinker", route_after_thinker, {
        "tool_executor": "tool_executor", "memory_writer": "memory_writer", "manager": "manager"
    })

    def route_after_tool_executor(state: PipelineState) -> str:
        # ... (same routing logic as V1.1.5, error routes to "manager")
        tool_result = state.get("current_tool_result"); error_message = state.get("error_message")
        tool_status = tool_result.get("status", "unknown") if isinstance(tool_result, dict) else "unknown"
        if error_message: return "manager" # Node error goes to manager
        # Tool failure or success, route back to thinker to process the result
        return "thinker"

    graph_builder.add_conditional_edges("tool_executor", route_after_tool_executor, {
        "thinker": "thinker", "manager": "manager"
    })

    graph_builder.add_edge("memory_writer", "manager") # After memory, go to manager
    graph_builder.add_edge("manager", END) # Manager is the final step before END

    try:
        compiled_graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info(f"create_rix_graph: Rix LangGraph compiled successfully using MemorySaver. Type: {type(compiled_graph)}")
        return compiled_graph
    except Exception as compile_e: logger.exception(f"create_rix_graph: FATAL: Failed to compile Rix LangGraph: {compile_e}"); return None

rix_graph_app: Optional[StateGraph] = None
def get_compiled_graph():
    global rix_graph_app; logger.info("get_compiled_graph: Called.")
    if rix_graph_app is None:
        logger.info("get_compiled_graph: rix_graph_app is None, attempting to create...")
        if LANGGRAPH_FULLY_LOADED and agent_nodes:
            rix_graph_app = create_rix_graph()
            if rix_graph_app: logger.info(f"get_compiled_graph: create_rix_graph SUCCEEDED. Type: {type(rix_graph_app)}")
            else: logger.error("get_compiled_graph: create_rix_graph FAILED and returned None.")
        else:
            if not LANGGRAPH_FULLY_LOADED: logger.error("get_compiled_graph: Cannot create, LANGGRAPH_FULLY_LOADED is False.")
            if not agent_nodes: logger.error("get_compiled_graph: Cannot create, agent_nodes is None.")
    else: logger.info("get_compiled_graph: rix_graph_app already exists.")
    return rix_graph_app

print(f"--- Module Defined: {__name__} (V1.1.6 - manager_node, MemorySaver) ---", flush=True)
# --- END OF FILE C:\Rix_Dev\Pro_Rix\Rix_Brain\orchestration\graph_pipeline.py ---