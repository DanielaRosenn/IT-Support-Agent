"""
LangGraph Workflow Definition

This file defines the main graph structure with all nodes and edges,
including autonomous re-routing between knowledge search and web search.
"""

from langgraph.graph import StateGraph, START, END
from src.graph.state import GraphState, GraphInput
from src.nodes.get_ticket_info import get_ticket_info
from src.nodes.search_memory import search_memory
from src.nodes.check_it_actions import check_it_actions
from src.nodes.extract_it_action_data import extract_it_action_data
from src.nodes.knowledge_search import knowledge_search
from src.nodes.evaluate_knowledge_sufficiency import evaluate_knowledge_sufficiency
from src.nodes.web_search_node import web_search_node
from src.nodes.extract_web_search_topics import extract_web_search_topics
from src.nodes.check_missing_information import check_missing_information
from src.nodes.augment_knowledge import augment_knowledge
from src.nodes.generate_ticket_response import generate_ticket_response
from src.nodes.evaluate_response import evaluate_response
from src.nodes.package_output import package_output


def route_after_it_action_check(state: GraphState) -> str:
    """
    Route after Step 3 based on IT action match result.

    If IT action matched: extract_it_action_data
    If no match: search_memory (new path)

    Args:
        state: Current graph state

    Returns:
        Next node name: "extract_it_action_data" or "search_memory"
    """
    is_match = state.get("is_it_action_match", False)

    if is_match:
        # IT Action matched - go to Step 4 to extract required data
        return "extract_it_action_data"
    else:
        # No IT Action match - go to memory/knowledge search
        return "search_memory"


def route_after_knowledge_evaluation(state: GraphState) -> str:
    """
    Route after knowledge sufficiency evaluation (Step 5b).

    If knowledge sufficient (score > 0.8): generate_ticket_response
    If knowledge insufficient (score ≤ 0.8): web_search_node

    Args:
        state: Current graph state

    Returns:
        Next node name: "generate_ticket_response" or "web_search_node"
    """
    knowledge_sufficiency = state.get("knowledge_sufficiency", {})
    is_sufficient = knowledge_sufficiency.get("is_sufficient", False)

    if is_sufficient:
        # Knowledge is sufficient - skip web search, go to response
        return "generate_ticket_response"
    else:
        # Knowledge insufficient - trigger web search
        return "web_search_node"


def route_after_web_search_topics(state: GraphState) -> str:
    """
    Route after web search topic extraction (Step 5d).

    If has topics: augment_knowledge (re-query KB)
    If no topics: check_missing_information (LLM gap analysis)

    Args:
        state: Current graph state

    Returns:
        Next node name: "augment_knowledge" or "check_missing_information"
    """
    web_topics = state.get("web_search_topics", {})
    has_topics = web_topics.get("has_topics", False)

    if has_topics:
        # Web search identified specific topics - augment KB with targeted queries
        return "augment_knowledge"
    else:
        # No specific topics - check if information is complete
        return "check_missing_information"


def route_after_missing_info_check(state: GraphState) -> str:
    """
    Route after missing information check (Step 6a).

    If needs augmentation AND under max iterations: augment_knowledge
    If complete OR max iterations reached: generate_ticket_response

    Args:
        state: Current graph state

    Returns:
        Next node name: "augment_knowledge" or "generate_ticket_response"
    """
    missing_info_check = state.get("missing_information_check", {})
    needs_augmentation = missing_info_check.get("needs_augmentation", False)

    if needs_augmentation:
        # More information needed - run targeted augmentation
        return "augment_knowledge"
    else:
        # Information complete or max iterations reached - generate response
        return "generate_ticket_response"


def route_after_augmentation(state: GraphState) -> str:
    """
    Route after knowledge augmentation (Step 6b).

    Always returns to check_missing_information for re-evaluation.
    This creates the iterative refinement loop (max 2 iterations).

    Args:
        state: Current graph state

    Returns:
        Next node name: "check_missing_information"
    """
    # Always loop back to check if augmentation was sufficient
    return "check_missing_information"


def create_graph():
    """
    Create and compile the LangGraph workflow with autonomous re-routing.

    Workflow structure:
    1. get_ticket_info
    2. check_it_actions
       - If match: extract_it_action_data → generate_ticket_response
       - If no match: search_memory → knowledge_search → evaluate_knowledge_sufficiency
    3. evaluate_knowledge_sufficiency
       - If sufficient: generate_ticket_response
       - If insufficient: web_search_node → extract_web_search_topics
    4. extract_web_search_topics
       - If has topics: augment_knowledge → check_missing_information
       - If no topics: check_missing_information
    5. check_missing_information
       - If needs augmentation: augment_knowledge (loop back, max 2 iterations)
       - If complete: generate_ticket_response

    Returns:
        Compiled graph ready for execution
    """
    # Create the graph with our state schema
    # input=GraphInput ensures only ticket_id is accepted as input
    workflow = StateGraph(GraphState, input=GraphInput)

    # ===== PHASE 1: NODES =====
    # Step 1: Get Ticket Info
    workflow.add_node("get_ticket_info", get_ticket_info)

    # Step 2: Search Confluence Memory (moved after IT action check)
    workflow.add_node("search_memory", search_memory)

    # Step 3: Check IT Actions Match
    workflow.add_node("check_it_actions", check_it_actions)

    # Step 4: Extract IT Action Data (IT action path only)
    workflow.add_node("extract_it_action_data", extract_it_action_data)

    # Step 5: Knowledge Search Pipeline
    workflow.add_node("knowledge_search", knowledge_search)

    # Step 5b: Evaluate Knowledge Sufficiency
    workflow.add_node("evaluate_knowledge_sufficiency", evaluate_knowledge_sufficiency)

    # Step 5c: Web Search (if insufficient knowledge)
    workflow.add_node("web_search_node", web_search_node)

    # Step 5d: Extract Web Search Topics
    workflow.add_node("extract_web_search_topics", extract_web_search_topics)

    # Step 6a: Check Missing Information
    workflow.add_node("check_missing_information", check_missing_information)

    # Step 6b: Augment Knowledge
    workflow.add_node("augment_knowledge", augment_knowledge)

    # Step 6: Generate Ticket Response
    workflow.add_node("generate_ticket_response", generate_ticket_response)

    # Step 7: Evaluate Response
    workflow.add_node("evaluate_response", evaluate_response)

    # Step 8: Package Output
    workflow.add_node("package_output", package_output)

    # ===== PHASE 2: EDGES =====

    # START → get_ticket_info
    workflow.add_edge(START, "get_ticket_info")

    # get_ticket_info → check_it_actions
    workflow.add_edge("get_ticket_info", "check_it_actions")

    # CONDITIONAL: check_it_actions → extract_it_action_data OR search_memory
    workflow.add_conditional_edges(
        "check_it_actions",
        route_after_it_action_check,
        {
            "extract_it_action_data": "extract_it_action_data",
            "search_memory": "search_memory"
        }
    )

    # IT action path: extract_it_action_data → generate_ticket_response
    workflow.add_edge("extract_it_action_data", "generate_ticket_response")

    # Knowledge search path: search_memory → knowledge_search
    workflow.add_edge("search_memory", "knowledge_search")

    # knowledge_search → evaluate_knowledge_sufficiency
    workflow.add_edge("knowledge_search", "evaluate_knowledge_sufficiency")

    # CONDITIONAL: evaluate_knowledge_sufficiency → generate_ticket_response OR web_search_node
    workflow.add_conditional_edges(
        "evaluate_knowledge_sufficiency",
        route_after_knowledge_evaluation,
        {
            "generate_ticket_response": "generate_ticket_response",
            "web_search_node": "web_search_node"
        }
    )

    # web_search_node → extract_web_search_topics
    workflow.add_edge("web_search_node", "extract_web_search_topics")

    # CONDITIONAL: extract_web_search_topics → augment_knowledge OR check_missing_information
    workflow.add_conditional_edges(
        "extract_web_search_topics",
        route_after_web_search_topics,
        {
            "augment_knowledge": "augment_knowledge",
            "check_missing_information": "check_missing_information"
        }
    )

    # CONDITIONAL: check_missing_information → augment_knowledge OR generate_ticket_response
    workflow.add_conditional_edges(
        "check_missing_information",
        route_after_missing_info_check,
        {
            "augment_knowledge": "augment_knowledge",
            "generate_ticket_response": "generate_ticket_response"
        }
    )

    # CONDITIONAL: augment_knowledge → check_missing_information (loop back)
    workflow.add_conditional_edges(
        "augment_knowledge",
        route_after_augmentation,
        {
            "check_missing_information": "check_missing_information"
        }
    )

    # generate_ticket_response → evaluate_response
    workflow.add_edge("generate_ticket_response", "evaluate_response")

    # evaluate_response → package_output
    workflow.add_edge("evaluate_response", "package_output")

    # package_output → END
    workflow.add_edge("package_output", END)

    # Compile the graph
    return workflow.compile()


# Create the graph instance
graph = create_graph()
