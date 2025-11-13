"""
Response Evaluation Node

Evaluates the quality of generated responses using LLM-based assessment.
Uses StaticEvaluationData extracted from GraphState (no additional tool calls).
Returns ResponseEvaluation with scores, strengths, weaknesses, and improvement suggestions.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from src.graph.state import GraphState
from src.models.evaluation_models import (
    StaticEvaluationData,
    ResponseEvaluation,
    TicketSnapshot,
    ResponseSnapshot,
    KnowledgeSourceSnapshot
)
from src.utils.llm_service import get_llm_service
from src.utils.prompt_utils import PromptBuilder, log_llm_response, parse_llm_json_response
from src.config.prompts import (
    RESPONSE_EVALUATOR_GUIDELINES,
    SELF_SERVICE_EVALUATION_CRITERIA,
    IT_EXECUTION_EVALUATION_CRITERIA,
    INVESTIGATION_EVALUATION_CRITERIA,
    EVALUATION_TASK,
    EVALUATION_JSON_FORMAT,
    JSON_INSTRUCTIONS
)
from src.utils.exceptions import LLMError, RecoverableError

logger = logging.getLogger(__name__)


# ============================================================================
# STATIC DATA EXTRACTION - Extract evaluation data from GraphState
# ============================================================================

def _extract_ticket_snapshot(state: GraphState) -> TicketSnapshot:
    """Extract ticket information snapshot from state."""
    ticket_info = state.get("ticket_info")

    if not ticket_info:
        raise ValueError("ticket_info missing from state")

    return TicketSnapshot(
        ticket_id=ticket_info.ticket_id,
        description=ticket_info.description,
        category=getattr(ticket_info, "category", None),
        subject=getattr(ticket_info, "subject", None),
        requester=getattr(ticket_info, "requester", None)
    )


def _extract_response_snapshot(state: GraphState) -> Tuple[ResponseSnapshot, str]:
    """
    Extract response snapshot from state.

    Returns:
        Tuple of (ResponseSnapshot, response_type_key) where response_type_key is
        "ticket_response", "it_solution_steps", or "it_further_investigation_actions"
    """
    # Determine which response field is populated
    response_content = None
    response_type = None
    response_type_key = None

    if state.get("ticket_response"):
        response_content = state["ticket_response"]
        response_type = "self_service"
        response_type_key = "ticket_response"
    elif state.get("it_solution_steps"):
        response_content = state["it_solution_steps"]
        response_type = "it_execution"
        response_type_key = "it_solution_steps"
    elif state.get("it_further_investigation_actions"):
        response_content = state["it_further_investigation_actions"]
        response_type = "investigation"
        response_type_key = "it_further_investigation_actions"
    else:
        raise ValueError("No response content found in state (ticket_response, it_solution_steps, or it_further_investigation_actions)")

    # Analyze response structure
    has_numbered_steps = bool(re.search(r'^\s*\d+[\.\)]\s+', response_content, re.MULTILINE))
    step_matches = re.findall(r'^\s*\d+[\.\)]\s+', response_content, re.MULTILINE)
    step_count = len(step_matches) if step_matches else None

    snapshot = ResponseSnapshot(
        response_type=response_type,
        content=response_content,
        response_length=len(response_content),
        has_numbered_steps=has_numbered_steps,
        step_count=step_count
    )

    return snapshot, response_type_key


def _extract_knowledge_sources(state: GraphState) -> List[KnowledgeSourceSnapshot]:
    """Extract knowledge sources from state (memory, articles, context grounding, web)."""
    sources = []

    # 1. Memory results (Confluence)
    memory_results = state.get("memory_results", [])
    for mem in memory_results:
        sources.append(KnowledgeSourceSnapshot(
            source_type="memory",
            content=mem.get("content", "")[:500],  # Truncate to 500 chars for prompt
            title=mem.get("title", None),
            score=mem.get("score", None),
            url=mem.get("url", None)
        ))

    # 2. FreshService articles
    freshservice_articles = state.get("freshservice_articles", [])
    for article in freshservice_articles:
        # Check if it's a dict or has attributes
        if isinstance(article, dict):
            content = article.get("description", "")
            title = article.get("title", None)
        else:
            content = getattr(article, "description", "")
            title = getattr(article, "title", None)

        sources.append(KnowledgeSourceSnapshot(
            source_type="freshservice",
            content=content[:500] if content else "",
            title=title,
            score=None,  # Articles don't have scores by default
            url=None
        ))

    # 3. Context Grounding results
    cg_results = state.get("context_grounding_results", [])
    for cg in cg_results:
        sources.append(KnowledgeSourceSnapshot(
            source_type="context_grounding",
            content=cg.get("content", "")[:500],
            title=cg.get("title", None),
            score=cg.get("score", None),
            url=cg.get("source", None)
        ))

    # 4. Web search results
    web_search_resolution = state.get("web_search_resolution", {})
    if web_search_resolution:
        # Web search has resolution_steps and sources
        resolution_steps = web_search_resolution.get("resolution_steps", [])
        web_sources = web_search_resolution.get("sources", [])

        if resolution_steps:
            sources.append(KnowledgeSourceSnapshot(
                source_type="web_search",
                content="\n".join(resolution_steps)[:500],
                title="Web Search Resolution",
                score=web_search_resolution.get("confidence", None),
                url=", ".join(web_sources[:3]) if web_sources else None
            ))

    return sources


def _extract_static_evaluation_data(state: GraphState) -> StaticEvaluationData:
    """
    Extract all static evaluation data from GraphState.

    This is the main function that builds the StaticEvaluationData object
    that will be passed to the LLM for evaluation.
    """
    logger.info("[Evaluation] Extracting static evaluation data from state")

    # Extract ticket snapshot
    ticket_snapshot = _extract_ticket_snapshot(state)

    # Extract response snapshot
    response_snapshot, response_type_key = _extract_response_snapshot(state)

    # Extract knowledge sources
    knowledge_sources = _extract_knowledge_sources(state)

    # Extract knowledge sufficiency metrics
    knowledge_sufficiency = state.get("knowledge_sufficiency", {})
    knowledge_sufficiency_score = knowledge_sufficiency.get("best_score", None)
    knowledge_was_sufficient = knowledge_sufficiency.get("is_sufficient", False)

    # Extract augmentation tracking
    augmentation_iteration = state.get("augmentation_iteration", 0)
    augmentation_source = state.get("augmentation_source", None)
    web_search_used = bool(state.get("web_search_resolution"))

    # Extract IT action context
    is_it_action_match = state.get("is_it_action_match", False)
    it_action_match = state.get("it_action_match", {})
    matched_it_action_name = it_action_match.get("matched_action_name", None) if is_it_action_match else None
    it_action_confidence = it_action_match.get("confidence", None) if is_it_action_match else None

    # Extract memory context
    memory_results = state.get("memory_results", [])
    memory_results_count = len(memory_results)
    memory_best_score = max([m.get("score", 0) for m in memory_results], default=None) if memory_results else None

    # Build StaticEvaluationData
    static_data = StaticEvaluationData(
        ticket=ticket_snapshot,
        response=response_snapshot,
        knowledge_sources=knowledge_sources,
        knowledge_sources_count=len(knowledge_sources),
        knowledge_sufficiency_score=knowledge_sufficiency_score,
        knowledge_was_sufficient=knowledge_was_sufficient,
        augmentation_iterations=augmentation_iteration,
        augmentation_source=augmentation_source,
        web_search_used=web_search_used,
        was_it_action_match=is_it_action_match,
        matched_it_action_name=matched_it_action_name,
        it_action_confidence=it_action_confidence,
        memory_results_count=memory_results_count,
        memory_best_score=memory_best_score
    )

    logger.info(f"[Evaluation] Static data extracted:")
    logger.info(f"  - Response type: {response_snapshot.response_type}")
    logger.info(f"  - Response length: {response_snapshot.response_length} chars")
    logger.info(f"  - Knowledge sources: {len(knowledge_sources)}")
    logger.info(f"  - Knowledge sufficiency score: {knowledge_sufficiency_score}")
    logger.info(f"  - Augmentation iterations: {augmentation_iteration}")
    logger.info(f"  - Web search used: {web_search_used}")

    return static_data


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def _get_criteria_for_response_type(response_type: str) -> str:
    """Get evaluation criteria based on response type."""
    if response_type == "self_service":
        return SELF_SERVICE_EVALUATION_CRITERIA
    elif response_type == "it_execution":
        return IT_EXECUTION_EVALUATION_CRITERIA
    elif response_type == "investigation":
        return INVESTIGATION_EVALUATION_CRITERIA
    else:
        logger.warning(f"Unknown response type: {response_type}, defaulting to self_service criteria")
        return SELF_SERVICE_EVALUATION_CRITERIA


def _format_knowledge_sources(sources: List[KnowledgeSourceSnapshot]) -> str:
    """Format knowledge sources for prompt."""
    if not sources:
        return "No knowledge sources available."

    formatted = []
    for i, source in enumerate(sources, 1):
        score_str = f" (score: {source.score:.2f})" if source.score else ""
        title_str = f" - {source.title}" if source.title else ""
        url_str = f"\n  URL: {source.url}" if source.url else ""

        formatted.append(
            f"{i}. [{source.source_type}]{title_str}{score_str}\n"
            f"  {source.content[:300]}...{url_str}"
        )

    return "\n\n".join(formatted)


def _build_evaluation_prompt(static_data: StaticEvaluationData) -> str:
    """
    Build the complete evaluation prompt.

    Structure:
    1. Evaluation guidelines
    2. Response-type-specific criteria
    3. Ticket context
    4. Response to evaluate
    5. Knowledge sources
    6. Evaluation task
    7. JSON format
    """
    criteria = _get_criteria_for_response_type(static_data.response.response_type)

    # Build ticket context
    ticket_context = f"""**Ticket ID:** {static_data.ticket.ticket_id}
**Subject:** {static_data.ticket.subject or "N/A"}
**Category:** {static_data.ticket.category or "N/A"}
**Requester:** {static_data.ticket.requester or "N/A"}
**Description:**
{static_data.ticket.description}"""

    # Build response context
    response_context = f"""**Response Type:** {static_data.response.response_type}
**Response Length:** {static_data.response.response_length} characters
**Has Numbered Steps:** {"Yes" if static_data.response.has_numbered_steps else "No"}
**Step Count:** {static_data.response.step_count or "N/A"}

**Response Content:**
{static_data.response.content}"""

    # Build knowledge sources context
    knowledge_context = f"""**Knowledge Sources Count:** {static_data.knowledge_sources_count}
**Knowledge Sufficiency Score:** {static_data.knowledge_sufficiency_score or "N/A"}
**Knowledge Was Sufficient:** {"Yes" if static_data.knowledge_was_sufficient else "No"}
**Augmentation Iterations:** {static_data.augmentation_iterations}
**Web Search Used:** {"Yes" if static_data.web_search_used else "No"}
**IT Action Match:** {"Yes" if static_data.was_it_action_match else "No"}
{f"**Matched IT Action:** {static_data.matched_it_action_name}" if static_data.was_it_action_match else ""}
**Memory Results:** {static_data.memory_results_count} results
{f"**Memory Best Score:** {static_data.memory_best_score:.2f}" if static_data.memory_best_score else ""}

**Sources:**
{_format_knowledge_sources(static_data.knowledge_sources)}"""

    # Build prompt using PromptBuilder
    prompt = (
        PromptBuilder("RESPONSE QUALITY EVALUATION")
        .add_section("Evaluation Guidelines", RESPONSE_EVALUATOR_GUIDELINES)
        .add_section("Response Type Criteria", criteria)
        .add_section("Original Ticket", ticket_context)
        .add_section("Generated Response", response_context)
        .add_section("Knowledge Sources Used", knowledge_context)
        .add_section("Evaluation Task", EVALUATION_TASK)
        .add_section("Response Format", EVALUATION_JSON_FORMAT + "\n\n" + JSON_INSTRUCTIONS)
        .build()
    )

    return prompt


# ============================================================================
# LLM INVOCATION & PARSING
# ============================================================================

async def _invoke_evaluation_llm(prompt: str) -> Dict[str, Any]:
    """
    Invoke LLM to evaluate the response.

    Returns:
        Dict containing the ResponseEvaluation JSON
    """
    llm = get_llm_service()

    logger.info("[Evaluation] Invoking LLM for response evaluation")

    try:
        response = await llm.invoke(
            prompt=prompt,
            max_tokens=2000,  # Evaluation responses can be detailed
            temperature=0.3,  # Low temperature for consistent evaluation
            system_message="You are an expert quality assurance specialist evaluating IT support responses. Provide objective, calibrated scores and actionable feedback.",
            operation_name="evaluate_response"
        )

        log_llm_response(logger, response, "Evaluation Response")

        # Parse JSON response
        evaluation_json = parse_llm_json_response(response, logger, "evaluate_response")

        return evaluation_json

    except Exception as e:
        logger.error(f"[Evaluation] LLM invocation failed: {str(e)}")
        raise LLMError(f"Failed to invoke LLM for evaluation: {str(e)}")


def _validate_and_build_evaluation(evaluation_json: Dict[str, Any]) -> ResponseEvaluation:
    """
    Validate and build ResponseEvaluation model from LLM JSON.

    Args:
        evaluation_json: Raw JSON from LLM

    Returns:
        ResponseEvaluation model instance

    Raises:
        ValueError: If validation fails
    """
    try:
        evaluation = ResponseEvaluation(**evaluation_json)

        logger.info(f"[Evaluation] Validation successful:")
        logger.info(f"  - Overall score: {evaluation.overall_score:.2f}")
        logger.info(f"  - Quality score: {evaluation.quality_score:.2f}")
        logger.info(f"  - Completeness score: {evaluation.completeness_score:.2f}")
        logger.info(f"  - Confidence score: {evaluation.confidence_score:.2f}")
        logger.info(f"  - Confidence level: {evaluation.confidence_level}")
        logger.info(f"  - Strengths: {len(evaluation.strengths)}")
        logger.info(f"  - Weaknesses: {len(evaluation.weaknesses)}")
        logger.info(f"  - Improvement suggestions: {len(evaluation.improvement_suggestions)}")

        return evaluation

    except Exception as e:
        logger.error(f"[Evaluation] Validation failed: {str(e)}")
        logger.error(f"[Evaluation] Raw JSON: {json.dumps(evaluation_json, indent=2)}")
        raise ValueError(f"Failed to validate evaluation response: {str(e)}")


def _create_fallback_evaluation(static_data: StaticEvaluationData, error: Exception) -> Dict[str, Any]:
    """
    Create a fallback evaluation if LLM evaluation fails.

    Uses heuristic-based scoring as a safety net.
    """
    logger.warning(f"[Evaluation] Creating fallback evaluation due to: {str(error)}")

    # Heuristic scoring based on available metrics
    knowledge_score = static_data.knowledge_sufficiency_score or 0.5
    has_steps = static_data.response.has_numbered_steps
    step_count = static_data.response.step_count or 0
    response_length = static_data.response.response_length

    # Simple heuristic: longer responses with numbered steps and good knowledge score are better
    completeness_score = min(0.5 + (step_count * 0.05) + (0.1 if has_steps else 0), 1.0)
    quality_score = min(0.5 + (0.2 if response_length > 200 else 0) + (0.1 if has_steps else 0), 1.0)
    confidence_score = knowledge_score

    overall_score = (quality_score * 0.4 + completeness_score * 0.3 + confidence_score * 0.3)

    # Determine confidence level
    if overall_score >= 0.7 and knowledge_score >= 0.7:
        confidence_level = "medium"  # Never "high" for fallback
    elif overall_score >= 0.5:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    fallback = {
        "evaluated": True,
        "response_type": static_data.response.response_type,
        "quality_score": round(quality_score, 2),
        "completeness_score": round(completeness_score, 2),
        "confidence_score": round(confidence_score, 2),
        "overall_score": round(overall_score, 2),
        "clarity_score": round(quality_score, 2) if static_data.response.response_type == "self_service" else None,
        "actionability_score": round(quality_score, 2) if static_data.response.response_type == "it_execution" else None,
        "diagnostic_depth_score": round(completeness_score, 2) if static_data.response.response_type == "investigation" else None,
        "strengths": ["Response generated successfully (heuristic evaluation)"],
        "weaknesses": ["Evaluation failed - using fallback heuristic scoring"],
        "missing_elements": [],
        "knowledge_assessment": f"Knowledge sufficiency score: {knowledge_score:.2f}. Augmentation iterations: {static_data.augmentation_iterations}.",
        "improvement_suggestions": ["LLM evaluation failed - manual review recommended"],
        "confidence_level": confidence_level,
        "evaluation_notes": f"Fallback evaluation due to LLM error: {str(error)[:100]}"
    }

    return fallback


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

async def evaluate_response(state: GraphState) -> Dict[str, Any]:
    """
    Evaluate the quality of the generated response.

    This node:
    1. Extracts StaticEvaluationData from GraphState
    2. Builds evaluation prompt with response-type-specific criteria
    3. Invokes LLM to evaluate the response
    4. Parses and validates ResponseEvaluation
    5. Returns evaluation to state

    Args:
        state: Current graph state

    Returns:
        dict: Updated state with response_evaluation field
    """
    logger.info("=" * 80)
    logger.info("[STEP 8b] EVALUATE RESPONSE")
    logger.info("=" * 80)

    try:
        # 1. Extract static data from state
        static_data = _extract_static_evaluation_data(state)

        # 2. Build evaluation prompt
        prompt = _build_evaluation_prompt(static_data)

        logger.info(f"[Evaluation] Prompt built: {len(prompt)} characters")

        # 3. Invoke LLM
        evaluation_json = await _invoke_evaluation_llm(prompt)

        # 4. Validate and build ResponseEvaluation model
        evaluation = _validate_and_build_evaluation(evaluation_json)

        # 5. Convert to dict for state
        evaluation_dict = evaluation.model_dump()

        logger.info("[Evaluation] Response evaluation completed successfully")
        logger.info("=" * 80)

        return {"response_evaluation": evaluation_dict}

    except (LLMError, ValueError) as e:
        # Recoverable error - create fallback evaluation
        logger.warning(f"[Evaluation] Evaluation failed, using fallback: {str(e)}")

        try:
            static_data = _extract_static_evaluation_data(state)
            fallback_evaluation = _create_fallback_evaluation(static_data, e)

            logger.info("[Evaluation] Fallback evaluation created")
            logger.info("=" * 80)

            return {"response_evaluation": fallback_evaluation}

        except Exception as fallback_error:
            logger.error(f"[Evaluation] Fallback evaluation also failed: {str(fallback_error)}")

            # Last resort: minimal evaluation
            minimal_evaluation = {
                "evaluated": False,
                "response_type": "unknown",
                "quality_score": 0.5,
                "completeness_score": 0.5,
                "confidence_score": 0.5,
                "overall_score": 0.5,
                "clarity_score": None,
                "actionability_score": None,
                "diagnostic_depth_score": None,
                "strengths": [],
                "weaknesses": ["Evaluation failed"],
                "missing_elements": [],
                "knowledge_assessment": "Evaluation not performed",
                "improvement_suggestions": [],
                "confidence_level": "low",
                "evaluation_notes": f"Evaluation failed: {str(e)}"
            }

            logger.info("[Evaluation] Minimal evaluation returned")
            logger.info("=" * 80)

            return {"response_evaluation": minimal_evaluation}

    except Exception as e:
        # Unexpected error - log but don't fail workflow
        logger.error(f"[Evaluation] Unexpected error during evaluation: {str(e)}", exc_info=True)

        # Return minimal evaluation
        minimal_evaluation = {
            "evaluated": False,
            "response_type": "unknown",
            "quality_score": 0.5,
            "completeness_score": 0.5,
            "confidence_score": 0.5,
            "overall_score": 0.5,
            "clarity_score": None,
            "actionability_score": None,
            "diagnostic_depth_score": None,
            "strengths": [],
            "weaknesses": ["Unexpected evaluation error"],
            "missing_elements": [],
            "knowledge_assessment": "Evaluation not performed due to error",
            "improvement_suggestions": [],
            "confidence_level": "low",
            "evaluation_notes": f"Unexpected error: {str(e)[:100]}"
        }

        logger.info("[Evaluation] Minimal evaluation returned after unexpected error")
        logger.info("=" * 80)

        return {"response_evaluation": minimal_evaluation}
