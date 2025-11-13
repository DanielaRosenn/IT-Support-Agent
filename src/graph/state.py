"""
LangGraph State Schema for IT Support Agent

This module defines the state that flows through the agent graph.
Each node reads from and updates this state.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import NotRequired, TypedDict

# Import TicketInfo model (dynamically created from config.UIPATH_OUTPUT_MAPPING)
from src.integrations.uipath_get_job_data import TicketInfo


class GraphInput(TypedDict):
    """
    Input schema for the agent - only accepts ticket_id.
    All other fields in GraphState are populated internally during execution.
    """
    ticket_id: str


class GraphState(TypedDict):
    """
    Main state object that flows through the LangGraph.

    Fields are added progressively as the agent moves through steps:
    - Step 1: ticket_info (TicketInfo - from config.UIPATH_OUTPUT_MAPPING)
    - Step 2: memory_results
    - Step 3: it_action_match
    - Step 4: it_action_data (if IT action matched)
    - Step 5: freshservice_articles, context_grounding_results, knowledge_sufficiency
    - Step 7: web_search_results, claude_knowledge, final_sufficiency
    - Step 8: draft_response
    - Step 8b: response_evaluation
    - Step 9: final_output
    - Step 10: llm_costs
    """

    # ===== INPUT =====
    ticket_id: str

    # ===== STEP 1: Get Ticket Information (REQUIRED) =====
    # Uses TicketInfo model dynamically created from config.UIPATH_OUTPUT_MAPPING
    # Required because we can't process without ticket info
    ticket_info: TicketInfo

    # ===== STEP 2: Search Confluence Memory =====
    memory_results: NotRequired[List[Dict[str, Any]]]

    # ===== STEP 3: Check IT Actions Match (REQUIRED) =====
    it_action_match: NotRequired[Dict[str, Any]]  # {is_match, matched_action_name, confidence, reasoning, ticket_information_required}
    # Required flag to determine routing - always set to True or False
    is_it_action_match: bool

    # ===== STEP 4: Extract IT Action Data (only if is_it_action_match=True) =====
    ticket_information: NotRequired[Dict[str, Any]]  # Extracted ticket information fields (matches ticket_information_required from it_actions.json)
    missing_information: NotRequired[List[str]]  # List of field names that couldn't be extracted
    clarifying_questions: NotRequired[List[str]]  # Questions to ask user for missing information
    can_proceed_with_it_action: NotRequired[bool]  # True if all required information extracted

    # ===== STEP 5: Knowledge Search Pipeline =====
    freshservice_articles: NotRequired[List[Dict[str, Any]]]
    context_grounding_results: NotRequired[List[Dict[str, Any]]]
    knowledge_sufficiency: NotRequired[Dict[str, Any]]
    # Structure: {"is_sufficient": bool, "best_score": float, "best_source": str}

    # ===== STEP 5b: Web Search Resolution =====
    web_search_resolution: NotRequired[Dict[str, Any]]
    # Structure: {"resolution_steps": List[str], "sources": List[str],
    #             "search_query_used": str, "confidence": float}

    # ===== STEP 5c: Web Search Topic Extraction =====
    web_search_topics: NotRequired[Dict[str, Any]]
    # Structure: {"has_topics": bool, "topics": List[str],
    #             "targeted_kb_queries": List[str], "reasoning": str}

    # ===== STEP 6: Missing Information Check & Augmentation =====
    missing_information_check: NotRequired[Dict[str, Any]]
    # Structure: {"needs_augmentation": bool, "missing_topics": List[str],
    #             "targeted_queries": List[str], "reasoning": str}

    augmentation_iteration: NotRequired[int]  # Track iterations (max 2)
    augmentation_source: NotRequired[str]  # "web_topics" | "missing_info"

    # ===== STEP 7: Fallback Research (Legacy - kept for compatibility) =====
    web_search_results: NotRequired[List[Dict[str, Any]]]
    claude_knowledge: NotRequired[Dict[str, Any]]  # Not implemented yet
    final_sufficiency: NotRequired[Dict[str, Any]]

    # ===== STEP 8: Generate Draft Response =====
    draft_response: NotRequired[Dict[str, Any]]

    # ===== STEP 8b: Response Evaluation =====
    response_evaluation: NotRequired[Dict[str, Any]]
    # Structure: ResponseEvaluation model (see src/models/evaluation_models.py)
    # {
    #   "evaluated": bool,
    #   "response_type": "self_service" | "it_execution" | "investigation",
    #   "quality_score": float (0.0-1.0),
    #   "completeness_score": float (0.0-1.0),
    #   "confidence_score": float (0.0-1.0),
    #   "overall_score": float (0.0-1.0),
    #   "clarity_score": Optional[float],  # For self_service
    #   "actionability_score": Optional[float],  # For it_execution
    #   "diagnostic_depth_score": Optional[float],  # For investigation
    #   "strengths": List[str],
    #   "weaknesses": List[str],
    #   "missing_elements": List[str],
    #   "knowledge_assessment": str,
    #   "improvement_suggestions": List[str],
    #   "confidence_level": "high" | "medium" | "low",
    #   "evaluation_notes": Optional[str]
    # }

    # ===== STEP 9: Ticket Response Generation =====
    # One of these three will be populated based on scenario:
    ticket_response: NotRequired[str]  # Self-service or clarifying questions (client-facing)
    it_solution_steps: NotRequired[str]  # IT team execution steps (internal)
    it_further_investigation_actions: NotRequired[str]  # Investigation steps (internal)

    # ===== STEP 10: Final Output (one of these will be populated) =====
    final_output: NotRequired[Dict[str, Any]]

    # ===== STEP 11: Cost Tracking (REQUIRED) =====
    # Required because we always calculate costs based on model usage
    llm_costs: Dict[str, Any]

    # ===== METADATA =====
    metadata: NotRequired[Dict[str, Any]]


class GraphOutput(TypedDict):
    """
    Final output structure returned by the agent.
    This matches the output structure defined in implementation_plan.md
    """

    # Ticket information (uses same TicketInfo from config)
    ticket_info: TicketInfo

    # Ticket analysis (generated during processing)
    ticket_analysis: Dict[str, Any]

    # Decision making
    decision: Dict[str, Any]

    # Response evaluation & scoring
    response_evaluation: Dict[str, Any]

    # Client response (if self_service)
    client_response: Optional[str]

    # IT execution instructions (if it_execution)
    it_execution_instructions: Optional[Dict[str, Any]]

    # IT investigation instructions (if it_investigation)
    it_investigation_instructions: Optional[Dict[str, Any]]

    # Solution sources
    solution_sources: Dict[str, Any]

    # LLM costs
    llm_costs: Dict[str, Any]

    # Metadata
    metadata: Dict[str, Any]