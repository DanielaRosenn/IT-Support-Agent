"""
Step 7: Package Output

Final node that transforms GraphState into structured GraphOutput.
This node runs AFTER generate_ticket_response and creates the final output structure
that matches the ticket resolution format (e.g., ticket_27421_password_reset.md).

Key responsibilities:
1. Determine solution type (Self-Service, IT Execution, Investigation)
2. Aggregate all sources from memory, knowledge base, and web search
3. Extract structured solution steps from prose response
4. Generate keywords and metadata
5. Build complete GraphOutput structure
"""

import re
import logging
from typing import Dict, Any, List, Optional
from src.graph.state import GraphState
from src.utils.keyword_extraction import extract_keywords_with_llm
from src.utils.llm_service import get_llm_service

logger = logging.getLogger(__name__)


async def package_output(state: GraphState) -> dict:
    """
    Package all state data into structured GraphOutput.

    This is the final transformation step that consolidates scattered
    state data into a clean, structured output format.

    Args:
        state: Current graph state with all accumulated data

    Returns:
        dict: Updated state with populated GraphOutput fields
    """
    logger.info("[Step 7] Packaging output into structured format")

    ticket_info = state["ticket_info"]

    try:
        # [1] Determine which response type was generated
        solution_type, response_text = _determine_solution_type(state)
        logger.info(f"[Step 7] Solution type: {solution_type}")

        # [2] Aggregate all sources
        solution_sources = _aggregate_sources(state)
        logger.info(
            f"[Step 7] Aggregated {solution_sources['summary']['total_sources']} sources: "
            f"{solution_sources['memory']['count']} memory, "
            f"{solution_sources['articles']['count']} articles, "
            f"{solution_sources['knowledge']['count']} knowledge, "
            f"{solution_sources['web']['count']} web"
        )

        # [3] Extract solution steps from response text
        solution_steps = _extract_solution_steps(response_text)
        logger.info(f"[Step 7] Extracted {len(solution_steps)} solution steps")

        # [4] Build ticket analysis with keywords
        ticket_analysis = await _build_ticket_analysis(
            state=state,
            solution_type=solution_type,
            response_text=response_text
        )
        logger.info(f"[Step 7] Generated ticket analysis with {len(ticket_analysis.get('keywords', []))} keywords")

        # [5] Build decision object from IT action match
        decision = _build_decision_object(state)

        # [6] Build response evaluation (placeholder for now)
        response_evaluation = _build_response_evaluation(state)

        # [7] Determine which response field to populate
        client_response = None
        it_execution_instructions = None
        it_investigation_instructions = None

        if solution_type == "Self-Service":
            client_response = response_text
        elif solution_type == "IT Execution Required":
            it_execution_instructions = {
                "instructions": response_text,
                "steps": solution_steps,
                "matched_action": decision.get("matched_action"),
                "extracted_information": state.get("ticket_information", {})
            }
        elif solution_type == "Further Investigation Needed":
            it_investigation_instructions = {
                "investigation_plan": response_text,
                "steps": solution_steps,
                "knowledge_gaps": _identify_knowledge_gaps(state)
            }

        # [8] Build metadata
        metadata = _build_metadata(state, solution_type)

        # [9] Collect LLM costs from service
        try:
            llm_service = get_llm_service()
            llm_costs = llm_service.get_total_costs()
            logger.info(
                f"[Step 7] LLM Costs: {llm_costs['llm_calls_count']} calls, "
                f"{llm_costs['total_tokens']:,} tokens, "
                f"${llm_costs['estimated_cost_usd']:.4f}"
            )
        except Exception as e:
            logger.warning(f"[Step 7] Failed to collect LLM costs: {e}")
            llm_costs = {
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "error": f"Failed to collect costs: {str(e)}"
            }

        logger.info("[Step 7] Output packaging complete")

        # Return the populated GraphOutput structure
        # Note: We return individual fields, not nested in a dict,
        # because LangGraph will merge them into state
        return {
            "ticket_info": ticket_info,
            "ticket_analysis": ticket_analysis,
            "decision": decision,
            "response_evaluation": response_evaluation,
            "client_response": client_response,
            "it_execution_instructions": it_execution_instructions,
            "it_investigation_instructions": it_investigation_instructions,
            "solution_sources": solution_sources,
            "solution_steps": solution_steps,  # Add to state for convenience
            "llm_costs": llm_costs,
            "metadata": metadata
        }

    except Exception as e:
        logger.error(f"[Step 7] Error packaging output: {str(e)}", exc_info=True)
        # Return minimal valid output on error
        return {
            "ticket_info": ticket_info,
            "ticket_analysis": {"error": f"Failed to package output: {str(e)}"},
            "decision": {},
            "response_evaluation": {},
            "client_response": None,
            "it_execution_instructions": None,
            "it_investigation_instructions": None,
            "solution_sources": {},
            "llm_costs": {},
            "metadata": {"error": True, "error_message": str(e)}
        }


def _determine_solution_type(state: GraphState) -> tuple[str, str]:
    """
    Determine solution type based on which response field is populated.

    Args:
        state: Current graph state

    Returns:
        Tuple of (solution_type, response_text)
        solution_type: "Self-Service" | "IT Execution Required" | "Further Investigation Needed"
        response_text: The actual response content
    """
    # Check which response field is populated
    ticket_response = state.get("ticket_response")
    it_solution_steps = state.get("it_solution_steps")
    it_investigation = state.get("it_further_investigation_actions")

    if ticket_response:
        # Self-service response (could be instructions or clarifying questions)
        clarifying_questions = state.get("clarifying_questions", [])
        if clarifying_questions:
            return "Clarifying Questions", ticket_response
        return "Self-Service", ticket_response

    elif it_solution_steps:
        # IT execution required
        return "IT Execution Required", it_solution_steps

    elif it_investigation:
        # Further investigation needed
        # it_investigation might be an InvestigationResponse object, convert to JSON string
        if hasattr(it_investigation, 'model_dump_json'):
            return "Further Investigation Needed", it_investigation.model_dump_json(indent=2)
        elif isinstance(it_investigation, dict):
            import json
            return "Further Investigation Needed", json.dumps(it_investigation, indent=2)
        else:
            return "Further Investigation Needed", str(it_investigation)

    else:
        # Fallback - should not happen
        logger.warning("[Step 7] No response field populated in state")
        return "Unknown", "No response generated"


def _aggregate_sources(state: GraphState) -> Dict[str, Any]:
    """
    Aggregate all sources from memory, articles (FreshService), knowledge base (Context Grounding), and web search.

    Returns a structured object with clear separation:
    - memory: Previous tickets from Confluence memory search
    - articles: Articles from FreshService
    - knowledge: Internal knowledge base from Context Grounding
    - web: Web search results
    - summary: Overall statistics

    Args:
        state: Current graph state

    Returns:
        Dict with sources organized by type with clear separation
    """
    # Initialize structured sources object
    sources = {
        "memory": {
            "count": 0,
            "sources": []
        },
        "articles": {
            "count": 0,
            "sources": []
        },
        "knowledge": {
            "count": 0,
            "sources": []
        },
        "web": {
            "count": 0,
            "sources": []
        },
        "summary": {
            "total_sources": 0,
            "primary_source_used": None,
            "all_systems_used": []
        }
    }

    # [1] Memory results (Confluence previous tickets)
    memory_results = state.get("memory_results", [])
    if memory_results:
        for result in memory_results[:5]:  # Top 5
            source_entry = {
                "title": result.get("source", "Previous Ticket"),
                "content_preview": result.get("content", "")[:200],
                "score": result.get("score", 0.0),
                "source_system": "confluence_memory",
                "metadata": result.get("metadata", {})
            }
            sources["memory"]["sources"].append(source_entry)

        sources["memory"]["count"] = len(sources["memory"]["sources"])
        sources["summary"]["all_systems_used"].append("memory")

    # [2] FreshService articles
    freshservice_articles = state.get("freshservice_articles", [])
    if freshservice_articles:
        for article in freshservice_articles[:5]:  # Top 5
            # Handle both Pydantic models and dicts
            if hasattr(article, 'source'):
                # Pydantic model - access attributes directly with fallback
                title = article.source if article.source is not None else 'FreshService Article'
                content_preview = (article.content[:200] if article.content else '')
                score = article.score if article.score is not None else 0.0
                metadata = article.metadata if article.metadata is not None else {}
            else:
                # Dictionary
                title = article.get("source", "FreshService Article")
                content_preview = article.get("content", "")[:200]
                score = article.get("score", 0.0)
                metadata = article.get("metadata", {})

            source_entry = {
                "title": title,
                "content_preview": content_preview,
                "score": score,
                "source_system": "freshservice",
                "metadata": metadata
            }
            sources["articles"]["sources"].append(source_entry)

        sources["articles"]["count"] = len(sources["articles"]["sources"])
        sources["summary"]["all_systems_used"].append("articles")

    # [3] Context Grounding results (Internal knowledge base)
    kb_results = state.get("context_grounding_results", [])
    if kb_results:
        for result in kb_results[:5]:  # Top 5
            # Handle both Pydantic models and dicts
            if hasattr(result, 'source'):
                # Pydantic model - access attributes directly with fallback
                title = result.source if result.source is not None else 'KB Article'
                content_preview = (result.content[:200] if result.content else '')
                score = result.score if result.score is not None else 0.0
                metadata = result.metadata if result.metadata is not None else {}
            else:
                # Dictionary
                title = result.get("source", "KB Article")
                content_preview = result.get("content", "")[:200]
                score = result.get("score", 0.0)
                metadata = result.get("metadata", {})

            source_entry = {
                "title": title,
                "content_preview": content_preview,
                "score": score,
                "source_system": "context_grounding",
                "metadata": metadata
            }
            sources["knowledge"]["sources"].append(source_entry)

        sources["knowledge"]["count"] = len(sources["knowledge"]["sources"])
        sources["summary"]["all_systems_used"].append("knowledge")

    # [4] Web search results
    web_resolution = state.get("web_search_resolution", {})
    if web_resolution.get("sources"):
        for source_url in web_resolution["sources"][:5]:  # Top 5
            source_entry = {
                "url": source_url,
                "title": source_url,  # For consistency
                "confidence": web_resolution.get("confidence", 0.0),
                "source_system": "web_search",
                "metadata": {
                    "search_query": web_resolution.get("search_query_used", "")
                }
            }
            sources["web"]["sources"].append(source_entry)

        sources["web"]["count"] = len(sources["web"]["sources"])
        sources["summary"]["all_systems_used"].append("web")

    # Calculate total sources
    sources["summary"]["total_sources"] = (
        sources["memory"]["count"] +
        sources["articles"]["count"] +
        sources["knowledge"]["count"] +
        sources["web"]["count"]
    )

    # Determine primary source (based on which has highest score/count)
    primary_candidates = []
    if sources["knowledge"]["count"] > 0:
        primary_candidates.append(("knowledge", sources["knowledge"]["count"]))
    if sources["articles"]["count"] > 0:
        primary_candidates.append(("articles", sources["articles"]["count"]))
    if sources["memory"]["count"] > 0:
        primary_candidates.append(("memory", sources["memory"]["count"]))
    if sources["web"]["count"] > 0:
        primary_candidates.append(("web", sources["web"]["count"]))

    if primary_candidates:
        # Sort by count (descending) and take first
        primary_candidates.sort(key=lambda x: x[1], reverse=True)
        sources["summary"]["primary_source_used"] = primary_candidates[0][0]
    else:
        sources["summary"]["primary_source_used"] = "none"

    return sources


def _extract_solution_steps(response_text: str) -> List[str]:
    """
    Extract numbered solution steps from prose response using regex.

    Handles various numbered list formats:
    - "1. Step one"
    - "1) Step one"
    - "Step 1: Do this"
    - "**1.** Step one"

    Args:
        response_text: The response text containing steps

    Returns:
        List of extracted step strings (without numbering)
    """
    steps = []

    # Pattern 1: Standard numbered lists (1. or 1))
    # Matches: "1. Text" or "1) Text" or "**1.** Text"
    pattern1 = r'(?:^|\n)\s*\*?\*?(\d+)[\.\)]\s*\*?\*?\s*(.+?)(?=\n\s*\*?\*?\d+[\.\)]|\n\n|$)'
    matches1 = re.findall(pattern1, response_text, re.MULTILINE | re.DOTALL)

    if matches1:
        for num, step_text in matches1:
            # Clean up step text
            step_text = step_text.strip()
            # Remove any trailing markdown or extra whitespace
            step_text = re.sub(r'\s+', ' ', step_text)
            if step_text and len(step_text) > 3:  # Valid step
                steps.append(step_text)

    # Pattern 2: Step N: format
    # Matches: "Step 1: Do this" or "**Step 1:** Do this"
    if not steps:
        pattern2 = r'(?:^|\n)\s*\*?\*?Step\s+(\d+):?\s*\*?\*?\s*(.+?)(?=\n\s*\*?\*?Step\s+\d+|\n\n|$)'
        matches2 = re.findall(pattern2, response_text, re.MULTILINE | re.DOTALL | re.IGNORECASE)

        if matches2:
            for num, step_text in matches2:
                step_text = step_text.strip()
                step_text = re.sub(r'\s+', ' ', step_text)
                if step_text and len(step_text) > 3:
                    steps.append(step_text)

    # If no numbered steps found, try bullet points
    if not steps:
        pattern3 = r'(?:^|\n)\s*[-\*]\s+(.+?)(?=\n\s*[-\*]|\n\n|$)'
        matches3 = re.findall(pattern3, response_text, re.MULTILINE | re.DOTALL)

        if matches3:
            for step_text in matches3:
                step_text = step_text.strip()
                step_text = re.sub(r'\s+', ' ', step_text)
                if step_text and len(step_text) > 3:
                    steps.append(step_text)

    logger.debug(f"[Step 7] Extracted {len(steps)} steps from response")

    return steps


async def _build_ticket_analysis(
    state: GraphState,
    solution_type: str,
    response_text: str
) -> Dict[str, Any]:
    """
    Build ticket analysis section with keywords and categorization.

    Args:
        state: Current graph state
        solution_type: Determined solution type
        response_text: The generated response

    Returns:
        Dict with ticket analysis data
    """
    ticket_info = state["ticket_info"]

    # Generate keywords using existing utility
    try:
        # Combine ticket info and response for keyword extraction
        combined_text = f"{ticket_info.subject} {ticket_info.description} {response_text}"
        keywords_str = await extract_keywords_with_llm(
            description=combined_text,
            subject=ticket_info.subject,
            category=ticket_info.category,
            max_keywords=15
        )
        keywords = keywords_str.split()[:15]  # Max 15 keywords
    except Exception as e:
        logger.warning(f"[Step 7] Failed to extract keywords: {e}")
        # Fallback: use simple word extraction
        keywords = _fallback_keyword_extraction(ticket_info, response_text)

    # Generate problem summary (first 2-3 sentences from description)
    problem_summary = _generate_problem_summary(ticket_info)

    return {
        "problem_summary": problem_summary,
        "solution_type": solution_type,
        "category": ticket_info.category,
        "subject": ticket_info.subject,
        "keywords": keywords,
        "requester": ticket_info.requester,
        "requester_email": ticket_info.requester_email
    }


def _generate_problem_summary(ticket_info) -> str:
    """
    Generate a concise problem summary from ticket info.

    Args:
        ticket_info: TicketInfo object

    Returns:
        Summary string (1-2 sentences)
    """
    description = ticket_info.description or ""
    subject = ticket_info.subject or ""

    # Take first 2 sentences or first 150 chars
    sentences = re.split(r'[.!?]\s+', description)

    if len(sentences) >= 2:
        summary = f"{sentences[0]}. {sentences[1]}."
    elif sentences:
        summary = sentences[0] if sentences[0].endswith('.') else f"{sentences[0]}."
    else:
        summary = subject

    # Truncate if too long
    if len(summary) > 200:
        summary = summary[:197] + "..."

    return summary.strip()


def _fallback_keyword_extraction(ticket_info, response_text: str) -> List[str]:
    """
    Fallback keyword extraction if LLM fails.

    Args:
        ticket_info: TicketInfo object
        response_text: Response text

    Returns:
        List of keywords
    """
    # Simple extraction: take unique words from subject and description
    text = f"{ticket_info.subject} {ticket_info.description}".lower()

    # Remove common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her'}

    words = re.findall(r'\b\w+\b', text)
    keywords = [w for w in words if w not in stopwords and len(w) > 3]

    # Return unique keywords, max 10
    return list(dict.fromkeys(keywords))[:10]


def _build_decision_object(state: GraphState) -> Dict[str, Any]:
    """
    Build decision object from IT action match data.

    Args:
        state: Current graph state

    Returns:
        Dict with decision data
    """
    it_action_match = state.get("it_action_match", {})
    is_it_action = state.get("is_it_action_match", False)

    decision = {
        "is_it_action": is_it_action,
        "matched_action": it_action_match.get("matched_action_name"),
        "confidence": it_action_match.get("confidence", 0.0),
        "reasoning": it_action_match.get("reasoning", ""),
        "can_proceed": state.get("can_proceed_with_it_action", True)
    }

    # Add knowledge sufficiency if available
    knowledge_sufficiency = state.get("knowledge_sufficiency", {})
    if knowledge_sufficiency:
        decision["knowledge_sufficient"] = knowledge_sufficiency.get("is_sufficient", False)
        decision["knowledge_score"] = knowledge_sufficiency.get("best_score", 0.0)

    return decision


def _build_response_evaluation(state: GraphState) -> Dict[str, Any]:
    """
    Build response evaluation object from state.

    Extracts the response_evaluation populated by the evaluate_response node.
    If evaluation is not present, returns a placeholder.

    Args:
        state: Current graph state

    Returns:
        Dict with evaluation data (ResponseEvaluation model)
    """
    # Get evaluation from state (populated by evaluate_response node)
    response_evaluation = state.get("response_evaluation")

    if response_evaluation and response_evaluation.get("evaluated", False):
        # Evaluation was performed successfully
        return response_evaluation
    else:
        # Evaluation not performed or failed - return placeholder
        return {
            "evaluated": False,
            "response_type": "unknown",
            "quality_score": None,
            "completeness_score": None,
            "confidence_score": None,
            "overall_score": None,
            "clarity_score": None,
            "actionability_score": None,
            "diagnostic_depth_score": None,
            "strengths": [],
            "weaknesses": [],
            "missing_elements": [],
            "knowledge_assessment": "Evaluation not performed",
            "improvement_suggestions": [],
            "confidence_level": "unknown",
            "evaluation_notes": "Response evaluation was not performed or failed"
        }


def _identify_knowledge_gaps(state: GraphState) -> List[str]:
    """
    Identify knowledge gaps from investigation path.

    Args:
        state: Current graph state

    Returns:
        List of identified knowledge gaps
    """
    gaps = []

    # Check if augmentation was attempted
    augmentation_source = state.get("augmentation_source")
    if augmentation_source and augmentation_source != "none":
        gaps.append(f"Required augmentation from: {augmentation_source}")

    # Check missing information from IT action path
    missing_info = state.get("missing_information", [])
    if missing_info:
        gaps.extend([f"Missing field: {field}" for field in missing_info])

    # Check knowledge sufficiency
    knowledge_sufficiency = state.get("knowledge_sufficiency", {})
    if not knowledge_sufficiency.get("is_sufficient", True):
        gaps.append("Initial knowledge search insufficient")

    return gaps if gaps else ["No specific gaps identified"]


def _build_metadata(state: GraphState, solution_type: str) -> Dict[str, Any]:
    """
    Build metadata object with workflow information.

    Args:
        state: Current graph state
        solution_type: Determined solution type

    Returns:
        Dict with metadata
    """
    return {
        "solution_type": solution_type,
        "workflow_path": _determine_workflow_path(state),
        "augmentation_performed": state.get("augmentation_source") not in [None, "none"],
        "augmentation_iterations": state.get("augmentation_iteration", 0),
        "web_search_performed": bool(state.get("web_search_resolution")),
        "memory_search_performed": bool(state.get("memory_results")),
        "knowledge_search_performed": bool(state.get("context_grounding_results"))
    }


def _determine_workflow_path(state: GraphState) -> str:
    """
    Determine which workflow path was taken.

    Args:
        state: Current graph state

    Returns:
        Workflow path description
    """
    is_it_action = state.get("is_it_action_match", False)

    if is_it_action:
        can_proceed = state.get("can_proceed_with_it_action", False)
        if can_proceed:
            return "IT Action → Execution"
        else:
            return "IT Action → Clarification Needed"
    else:
        has_web = bool(state.get("web_search_resolution"))
        has_augmentation = state.get("augmentation_source") not in [None, "none"]

        if has_augmentation:
            return "Knowledge Search → Augmentation → Resolution"
        elif has_web:
            return "Knowledge Search → Web Search → Resolution"
        else:
            return "Knowledge Search → Direct Resolution"