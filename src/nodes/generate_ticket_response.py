"""
Step 6: Generate Ticket Response

Generates appropriate response based on ticket classification:
1. ticket_response: Self-service instructions or clarifying questions (client-facing)
2. it_solution_steps: IT team execution steps (internal)
3. it_further_investigation_actions: Investigation steps (internal)

Uses response_quality_guidelines.md principles for concise, action-oriented responses.
"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from src.graph.state import GraphState
from src.utils.llm_service import get_llm_service
from src.utils.prompt_utils import build_ticket_context, log_llm_response, PromptBuilder
from src.config.prompts import (
    CATO_AGENT_GUIDELINES,
    RESPONSE_TYPE_DECISION_TASK,
    RESPONSE_TYPE_DECISION_CRITERIA,
    RESPONSE_TYPE_DECISION_EXAMPLES,
    RESPONSE_TYPE_DECISION_FORMAT,
    SELF_SERVICE_RESPONSE_INSTRUCTIONS,
    SELF_SERVICE_RESPONSE_EXAMPLE,
    CLARIFYING_QUESTIONS_INSTRUCTIONS,
    CLARIFYING_QUESTIONS_EXAMPLE,
    IT_EXECUTION_STEPS_INSTRUCTIONS,
    IT_EXECUTION_FROM_KB_TASK,
    IT_EXECUTION_FROM_KB_TASK_DESCRIPTION,
    IT_EXECUTION_FROM_KB_FORMAT,
    INVESTIGATION_TASK,
    INVESTIGATION_ANALYSIS_TASK,
    INVESTIGATION_FORMAT
)
from src.utils.exceptions import LLMError, FatalError

logger = logging.getLogger(__name__)


# ============================================================================
# Response Models - Structured responses for investigation actions
# ============================================================================

class InitialAssessment(BaseModel):
    """Initial assessment of the issue for investigation."""
    issue_type: str = Field(description="Type of issue based on collected information")
    urgency_level: str = Field(description="High, Medium, or Low")
    complexity_estimate: str = Field(description="Simple, Moderate, or Complex")
    information_quality: str = Field(description="Complete, Partial, or Limited")


class InvestigationResponse(BaseModel):
    """
    Structured investigation response for IT team.

    Matches the format defined in INVESTIGATION_FORMAT prompt constant.
    Replaces string-based investigation response with validated structure.
    """
    issue_summary: str = Field(description="1-2 sentences summarizing the issue")
    initial_assessment: InitialAssessment = Field(description="Initial assessment of the issue")
    key_findings: List[str] = Field(description="2-3 key insights from memory/KB/web search")
    investigation_steps: List[str] = Field(description="Ordered list of diagnostic steps")
    knowledge_base_references: List[str] = Field(description="Most relevant sources to consult")
    estimated_resolution_time: str = Field(description="Time estimate with justification")
    recommended_next_actions: List[str] = Field(description="Immediate next steps for IT team")


async def generate_ticket_response(state: GraphState) -> dict:
    """
    Step 6: Generate appropriate response based on ticket classification.

    Response types:
    - ticket_response: For self-service or requesting more info (client-facing)
    - it_solution_steps: For IT team to execute (internal)
    - it_further_investigation_actions: For investigation (internal)

    Args:
        state: Current graph state

    Returns:
        dict: Updated state with one of the three response fields populated
    """
    ticket_info = state["ticket_info"]
    is_it_action_match = state.get("is_it_action_match", False)

    logger.info(f"[Step 6] Generating ticket response")
    logger.info(f"  - Ticket ID: {ticket_info.ticket_id}")
    logger.info(f"  - IT Action Match: {is_it_action_match}")

    try:
        if is_it_action_match:
            # IT Action path - check if we have all required info
            return await _generate_it_action_response(state)
        else:
            # Knowledge search path - determine if self-service or investigation
            return await _generate_knowledge_based_response(state)

    except LLMError as e:
        # LLM failed to generate response - this is critical for final response
        logger.error(
            f"[Step 6] LLM failed to generate ticket response: {e.message}",
            extra={"details": e.details, "ticket_id": ticket_info.ticket_id}
        )
        raise FatalError(
            "Failed to generate ticket response due to LLM error",
            details={"original_error": e.message, "ticket_id": ticket_info.ticket_id}
        )

    except Exception as e:
        # Unexpected error in response generation - fatal
        logger.error(
            f"[Step 6] Unexpected error generating ticket response: {str(e)}",
            exc_info=True,
            extra={"ticket_id": ticket_info.ticket_id}
        )
        raise FatalError(
            f"Unexpected error in ticket response generation: {str(e)}",
            details={"ticket_id": ticket_info.ticket_id}
        )


async def _generate_it_action_response(state: GraphState) -> Dict[str, Any]:
    """
    Generate response for IT Action match scenarios.

    Decision tree:
    - If can_proceed_with_it_action=False → ticket_response (clarifying questions)
    - If IT action is self-service → ticket_response (instructions)
    - If IT action requires IT team → it_solution_steps (IT instructions)
    """
    ticket_info = state["ticket_info"]
    it_action_match = state.get("it_action_match", {})
    matched_action_name = it_action_match.get("matched_action_name")
    llm_reasoning = it_action_match.get("reasoning", "")

    can_proceed = state.get("can_proceed_with_it_action", False)
    clarifying_questions = state.get("clarifying_questions", [])
    ticket_information = state.get("ticket_information", {})

    logger.info(f"[Step 6] IT Action response generation:")
    logger.info(f"  - Matched Action: {matched_action_name}")
    logger.info(f"  - Can Proceed: {can_proceed}")
    logger.info(f"  - Has Clarifying Questions: {len(clarifying_questions) > 0}")

    # Case 1: Need more information - send clarifying questions
    if not can_proceed and clarifying_questions:
        logger.info("[Step 6] Generating clarifying questions response")
        return {
            "ticket_response": await _generate_clarifying_questions_response(
                ticket_info=ticket_info,
                clarifying_questions=clarifying_questions,
                matched_action_name=matched_action_name,
                llm_reasoning=llm_reasoning
            )
        }

    # Case 2: Have all info - determine if self-service or IT execution
    # TODO: Add logic to check if action is self-service vs IT execution
    # For now, default to IT execution steps
    logger.info("[Step 6] Generating IT execution steps")
    return {
        "it_solution_steps": await _generate_it_execution_steps(
            ticket_info=ticket_info,
            matched_action_name=matched_action_name,
            llm_reasoning=llm_reasoning,
            ticket_information=ticket_information
        )
    }


async def _generate_knowledge_based_response(state: GraphState) -> Dict[str, Any]:
    """
    Generate response for knowledge-search path (non-IT-action tickets).

    Uses LLM to determine the appropriate response type (THREE-WAY DECISION):
    1. "user_executable" → ticket_response (client-facing self-service)
    2. "clear_it_instructions" → it_solution_steps (IT execution from KB)
    3. "requires_investigation" → it_further_investigation_actions (investigation)

    Decision logic:
    - User executable steps → ticket_response (self-service)
    - Clear IT instructions in KB → it_solution_steps (IT execution)
    - Unclear/incomplete → it_further_investigation_actions (investigation)

    Integrates ALL knowledge sources:
    - Confluence memory results
    - Context grounding results (including augmented)
    - Web search resolution
    - Augmentation metadata
    """
    ticket_info = state["ticket_info"]
    memory_results = state.get("memory_results", [])
    freshservice_articles = state.get("freshservice_articles", [])
    knowledge_results = state.get("context_grounding_results", [])
    web_resolution = state.get("web_search_resolution", {})
    knowledge_sufficiency = state.get("knowledge_sufficiency", {})
    augmentation_source = state.get("augmentation_source", None)
    augmentation_iteration = state.get("augmentation_iteration", 0)

    logger.info("[Step 6] Generating knowledge-based response")
    logger.info(f"  - Memory results: {len(memory_results)}")
    logger.info(f"  - FreshService articles: {len(freshservice_articles)}")
    logger.info(f"  - Knowledge results: {len(knowledge_results)}")
    logger.info(f"  - Web search available: {bool(web_resolution.get('resolution_steps'))}")
    logger.info(f"  - Augmentation: {augmentation_source} (iteration {augmentation_iteration})")

    # Combine FreshService articles and Context Grounding for unified view
    # FreshService articles are typically more actionable
    all_knowledge = freshservice_articles + knowledge_results

    # Use LLM to determine response type (three-way decision)
    response_type = await _determine_response_type(
        memory_results=memory_results,
        knowledge_results=all_knowledge,  # Now includes FreshService articles
        web_resolution=web_resolution
    )

    if response_type == "user_executable":
        logger.info("[Step 6] Generating self-service ticket response for user")
        return {
            "ticket_response": await _generate_self_service_response(
                ticket_info=ticket_info,
                memory_results=memory_results,
                knowledge_results=all_knowledge,  # Use combined articles
                web_resolution=web_resolution,
                knowledge_sufficiency=knowledge_sufficiency,
                augmentation_source=augmentation_source
            )
        }

    elif response_type == "clear_it_instructions":
        logger.info("[Step 6] Generating IT execution steps from knowledge base")
        return {
            "it_solution_steps": await _generate_it_execution_steps_from_knowledge(
                ticket_info=ticket_info,
                memory_results=memory_results,
                knowledge_results=all_knowledge,  # Use combined articles
                web_resolution=web_resolution,
                knowledge_sufficiency=knowledge_sufficiency
            )
        }

    else:  # requires_investigation
        logger.info("[Step 6] Generating IT investigation actions (insufficient for clear solution)")
        return {
            "it_further_investigation_actions": await _generate_investigation_steps(
                ticket_info=ticket_info,
                memory_results=memory_results,
                knowledge_results=all_knowledge,  # Use combined articles
                web_resolution=web_resolution,
                knowledge_sufficiency=knowledge_sufficiency,
                augmentation_source=augmentation_source,
                augmentation_iteration=augmentation_iteration
            )
        }


async def _determine_response_type(
    memory_results: list,
    knowledge_results: list,
    web_resolution: Dict[str, Any]
) -> str:
    """
    Use LLM to determine the appropriate response type based on collected knowledge.

    Three-way decision:
    1. "user_executable": User can perform steps themselves (VPN, restart, clear cache)
    2. "clear_it_instructions": Clear IT execution steps exist (admin actions, backend changes)
    3. "requires_investigation": Unclear/incomplete information, needs diagnostic investigation

    This is CONTENT-based analysis, not score-based. The LLM analyzes:
    - Whether steps are user-executable vs require IT/admin access
    - Whether clear, actionable instructions exist or investigation is needed
    - Quality and completeness of the solution content

    Returns one of: "user_executable", "clear_it_instructions", "requires_investigation"
    """
    llm_service = get_llm_service()

    # Quick check: If no information at all, requires IT investigation
    if not memory_results and not knowledge_results and not web_resolution.get("resolution_steps"):
        logger.info(f"[Response Type Check] ✗ No information sources available → IT investigation")
        return "requires_investigation"

    logger.info(f"[Response Type Check] Analyzing content to determine response type...")

    # Extract FULL content from top sources for detailed LLM analysis
    # Prioritize high-scoring sources but include full text for content analysis
    sources_for_analysis = []

    # Add top 3 KB articles (FreshService + Context Grounding) with full content
    if knowledge_results:
        for i, result in enumerate(knowledge_results[:3], 1):
            # Handle both dict and Pydantic formats
            if hasattr(result, 'content'):
                content = getattr(result, 'content', '')
                title = getattr(result, 'title', getattr(result, 'source', 'Unknown'))
                score = getattr(result, 'relevance_score', getattr(result, 'score', 0.0))
            else:
                # FreshService articles: relevance_score (from re-ranking), description (not content)
                content = result.get('content', result.get('description', ''))
                title = result.get('title', result.get('source', 'Unknown'))
                score = result.get("relevance_score", result.get("semantic_score", result.get("score", 0.0)))

            sources_for_analysis.append(f"""
[ARTICLE {i}] {title} (relevance: {score:.2f})
{content[:1000]}
""")

    # Add top 2 memory results with full content
    # IMPORTANT: Memory results contain "## Solution Type" metadata (Self-Service, IT Action, Investigation)
    if memory_results:
        for i, result in enumerate(memory_results[:2], 1):
            content = result.get('content', '')
            source = result.get('source', 'Unknown')
            score = result.get('score', 0.0)

            sources_for_analysis.append(f"""
[MEMORY {i}] {source} (relevance: {score:.2f})
NOTE: This memory includes "## Solution Type" label - pay attention to it!
{content[:1000]}
""")

    # Add web resolution steps
    if web_resolution.get('resolution_steps'):
        steps = web_resolution.get('resolution_steps', [])
        steps_text = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(steps)])
        sources_for_analysis.append(f"""
[WEB SEARCH RESOLUTION]
{steps_text}
""")

    all_content = "\n".join(sources_for_analysis) if sources_for_analysis else "No content available"

    # Build prompt using centralized content
    prompt = (PromptBuilder("DETERMINE RESPONSE TYPE BASED ON SOLUTION CONTENT")
        .add_section(None, RESPONSE_TYPE_DECISION_TASK)
        .add_section("AVAILABLE SOLUTION CONTENT", all_content)
        .add_section("DECISION CRITERIA", RESPONSE_TYPE_DECISION_CRITERIA)
        .add_section("EXAMPLES", RESPONSE_TYPE_DECISION_EXAMPLES)
        .add_section("YOUR RESPONSE", RESPONSE_TYPE_DECISION_FORMAT)
        .set_response_format("text")
        .build())

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=30,
        temperature=0.1,
        operation_name="determine_response_type"
    )

    decision = response.strip().lower()
    logger.info(f"[Response Type Check] LLM decision: {decision}")

    # Parse response - three-way decision
    if "user_executable" in decision:
        logger.info(f"[Response Type Check] ✓ Solution is USER-EXECUTABLE → ticket_response")
        return "user_executable"
    elif "clear_it_instructions" in decision:
        logger.info(f"[Response Type Check] ✓ CLEAR IT INSTRUCTIONS found → it_solution_steps")
        return "clear_it_instructions"
    else:
        logger.info(f"[Response Type Check] ✗ REQUIRES INVESTIGATION → it_further_investigation_actions")
        return "requires_investigation"


async def _generate_self_service_response(
    ticket_info: Any,
    memory_results: list,
    knowledge_results: list,
    web_resolution: Dict[str, Any],
    knowledge_sufficiency: Dict[str, Any],
    augmentation_source: str = None
) -> str:
    """
    Generate self-service response for user based on collected knowledge.

    Integrates information from ALL sources and provides clear, actionable
    steps the user can perform themselves.

    Follows response_quality_guidelines.md patterns.
    """
    llm_service = get_llm_service()

    ticket_context = build_ticket_context(ticket_info)

    # Format memory results
    memory_text = "No previous similar tickets found."
    if memory_results:
        memory_items = []
        for i, r in enumerate(memory_results[:2], 1):
            content_preview = (r.content if hasattr(r, 'content') else r.get('content', ''))[:200]
            memory_items.append(f"  {i}. {content_preview}...")
        memory_text = "\n".join(memory_items)

    # Format knowledge base results
    kb_text = "No internal KB articles found."
    if knowledge_results:
        kb_items = []
        for i, r in enumerate(knowledge_results[:3], 1):
            content_preview = (r.content if hasattr(r, 'content') else r.get('content', ''))[:200]
            kb_items.append(f"  {i}. {content_preview}...")
        kb_text = "\n".join(kb_items)

    # Format web search results
    web_text = "No web search performed."
    if web_resolution.get("resolution_steps"):
        steps = web_resolution.get("resolution_steps", [])
        web_text = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(steps)])

    # Format collected information
    collected_info = f"""[1] CONFLUENCE MEMORY (Previous Similar Tickets):
{memory_text}

[2] INTERNAL KNOWLEDGE BASE:
{kb_text}

[3] WEB SEARCH RESOLUTION:
{web_text}"""

    # Build prompt using centralized content with custom response task
    response_task = f"""Synthesize the information above into a clear, actionable response for {ticket_info.requester}.

You are generating a CLIENT-FACING response to help the user resolve their issue.

{SELF_SERVICE_RESPONSE_INSTRUCTIONS}"""

    prompt = (PromptBuilder("GENERATE SELF-SERVICE TICKET RESPONSE")
        .add_section("TICKET CONTEXT", ticket_context)
        .add_section("COLLECTED INFORMATION FROM ALL SOURCES", collected_info)
        .add_section("YOUR RESPONSE TASK", response_task)
        .add_section("EXAMPLE", SELF_SERVICE_RESPONSE_EXAMPLE)
        .add_section("YOUR RESPONSE", "Generate the complete email response now:")
        .set_response_format("text")
        .build())

    logger.debug("Sending self-service response generation to LLM")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=600,
        temperature=0.3
    )

    log_llm_response(logger, response, "SELF-SERVICE RESPONSE")

    return response.strip()


async def _generate_clarifying_questions_response(
    ticket_info: Any,
    clarifying_questions: list,
    matched_action_name: str,
    llm_reasoning: str
) -> str:
    """
    Generate a client-facing response with clarifying questions.

    Follows response_quality_guidelines.md patterns:
    - Brief greeting
    - Acknowledge the issue
    - Present questions clearly
    - Offer follow-up help
    """
    llm_service = get_llm_service()

    # Build ticket context
    ticket_context = build_ticket_context(ticket_info)

    # Format questions
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(clarifying_questions)])

    # Format context information
    context_info = f"""TICKET CONTEXT:
{ticket_context}

IT ACTION CLASSIFIED: {matched_action_name}
LLM CLASSIFICATION REASONING:
{llm_reasoning}

CLARIFYING QUESTIONS TO ASK:
{questions_text}"""

    # Build prompt using centralized content
    response_task = f"""You are generating a response to a ticket requester to gather missing information.

{CLARIFYING_QUESTIONS_INSTRUCTIONS}"""

    prompt = (PromptBuilder("GENERATE CLARIFYING QUESTIONS RESPONSE")
        .add_section(None, context_info)
        .add_section("RESPONSE STYLE (CRITICAL - FOLLOW EXACTLY)", response_task)
        .add_section("EXAMPLE", CLARIFYING_QUESTIONS_EXAMPLE)
        .add_section("YOUR RESPONSE", "Generate the complete email response now:")
        .set_response_format("text")
        .build())

    logger.debug("Sending clarifying questions response generation to LLM")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=400,
        temperature=0.3
    )

    log_llm_response(logger, response, "CLARIFYING QUESTIONS RESPONSE")

    return response.strip()


async def _generate_it_execution_steps(
    ticket_info: Any,
    matched_action_name: str,
    llm_reasoning: str,
    ticket_information: Dict[str, Any]
) -> str:
    """
    Generate IT team execution steps (internal document).

    Includes:
    - Summary of issue
    - LLM reasoning for classification
    - Step-by-step execution instructions
    - Verification steps
    """
    llm_service = get_llm_service()

    ticket_context = build_ticket_context(ticket_info)

    # Format extracted information
    info_text = "\n".join([f"  - {key}: {value}" for key, value in ticket_information.items()])

    # Format context information
    context_info = f"""TICKET CONTEXT:
{ticket_context}

IT ACTION CLASSIFIED: {matched_action_name}

LLM CLASSIFICATION REASONING:
{llm_reasoning}

EXTRACTED TICKET INFORMATION:
{info_text}"""

    # Format instructions with dynamic values
    formatted_instructions = IT_EXECUTION_STEPS_INSTRUCTIONS.format(
        llm_reasoning=llm_reasoning,
        matched_action_name=matched_action_name
    )

    # Build prompt using centralized content
    prompt = (PromptBuilder("GENERATE IT EXECUTION STEPS")
        .add_section(None, CATO_AGENT_GUIDELINES)
        .add_section("YOUR TASK", IT_EXECUTION_STEPS_INSTRUCTIONS.split('\n\n')[0])  # First paragraph only
        .add_section(None, context_info)
        .add_section("RESPONSE FORMAT", formatted_instructions)
        .add_section("YOUR RESPONSE", "Generate the IT execution document now:")
        .set_response_format("text")
        .build())

    logger.debug("Sending IT execution steps generation to LLM")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=600,
        temperature=0.2
    )

    log_llm_response(logger, response, "IT EXECUTION STEPS")

    return response.strip()


async def _generate_it_execution_steps_from_knowledge(
    ticket_info: Any,
    memory_results: list,
    knowledge_results: list,
    web_resolution: Dict[str, Any],
    knowledge_sufficiency: Dict[str, Any]
) -> str:
    """
    Generate IT team execution steps from knowledge base (internal document).

    Similar to _generate_it_execution_steps but:
    - Source: Knowledge base articles/memory (not IT action match)
    - Synthesizes instructions from collected knowledge
    - Less certain than IT action match (medium confidence vs high)

    Used when knowledge base has clear IT instructions but didn't match predefined IT action.
    """
    llm_service = get_llm_service()

    ticket_context = build_ticket_context(ticket_info)

    # Format memory results
    memory_text = "No previous similar tickets found."
    if memory_results:
        memory_items = []
        for i, r in enumerate(memory_results[:2], 1):
            content_preview = (r.content if hasattr(r, 'content') else r.get('content', ''))[:300]
            memory_items.append(f"  {i}. {content_preview}...")
        memory_text = "\n".join(memory_items)

    # Format knowledge base results
    kb_text = "No internal KB articles found."
    if knowledge_results:
        kb_items = []
        for i, r in enumerate(knowledge_results[:3], 1):
            content_preview = (r.content if hasattr(r, 'content') else r.get('content', ''))[:300]
            kb_items.append(f"  {i}. {content_preview}...")
        kb_text = "\n".join(kb_items)

    # Format web search results
    web_text = "No web search performed."
    if web_resolution.get("resolution_steps"):
        steps = web_resolution.get("resolution_steps", [])
        web_text = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(steps)])

    # Format knowledge sources
    knowledge_sources = f"""[1] CONFLUENCE MEMORY (Previous Similar Tickets):
{memory_text}

[2] INTERNAL KNOWLEDGE BASE:
{kb_text}

[3] WEB SEARCH RESOLUTION:
{web_text}"""

    # Build prompt using centralized content
    prompt = (PromptBuilder("GENERATE IT EXECUTION STEPS FROM KNOWLEDGE BASE")
        .add_section(None, CATO_AGENT_GUIDELINES)
        .add_section("YOUR TASK", IT_EXECUTION_FROM_KB_TASK)
        .add_section("TICKET CONTEXT", ticket_context)
        .add_section("KNOWLEDGE BASE SOURCES", knowledge_sources)
        .add_section("YOUR TASK", IT_EXECUTION_FROM_KB_TASK_DESCRIPTION)
        .add_section("RESPONSE FORMAT", IT_EXECUTION_FROM_KB_FORMAT)
        .add_section("YOUR RESPONSE", "Generate the IT execution document now:")
        .set_response_format("text")
        .build())

    logger.debug("Sending IT execution steps from knowledge generation to LLM")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=700,
        temperature=0.2,
        operation_name="generate_it_execution_steps_from_knowledge"
    )

    log_llm_response(logger, response, "IT EXECUTION STEPS (FROM KNOWLEDGE)")

    return response.strip()


async def _generate_investigation_steps(
    ticket_info: Any,
    memory_results: list,
    knowledge_results: list,
    web_resolution: Dict[str, Any],
    knowledge_sufficiency: Dict[str, Any],
    augmentation_source: str = None,
    augmentation_iteration: int = 0
) -> InvestigationResponse:
    """
    Generate investigation actions for IT team (internal document).

    Returns structured InvestigationResponse object instead of string.

    Integrates ALL collected knowledge sources:
    - Confluence memory results
    - Context grounding results (including augmented)
    - Web search resolution with external sources
    - Knowledge sufficiency evaluation
    - Augmentation metadata
    """
    llm_service = get_llm_service()

    ticket_context = build_ticket_context(ticket_info)

    # Format Confluence memory results
    memory_text = "No Confluence memory results found."
    if memory_results:
        memory_items = []
        for i, r in enumerate(memory_results[:3], 1):
            content_preview = (r.content if hasattr(r, 'content') else r.get('content', ''))[:150]
            source = r.source if hasattr(r, 'source') else r.get('source', 'Unknown')
            score = r.score if hasattr(r, 'score') else r.get('score', 0)
            memory_items.append(
                f"  {i}. [{source}] Score: {score:.2f}\n"
                f"     Preview: {content_preview}..."
            )
        memory_text = "\n".join(memory_items)

    # Format Context Grounding results (including augmented)
    knowledge_text = "No Context Grounding results found."
    if knowledge_results:
        knowledge_items = []
        for i, r in enumerate(knowledge_results[:5], 1):
            content_preview = (r.content if hasattr(r, 'content') else r.get('content', ''))[:150]
            source = r.source if hasattr(r, 'source') else r.get('source', 'Unknown')
            score = r.score if hasattr(r, 'score') else r.get('score', 0)
            knowledge_items.append(
                f"  {i}. [{source}] Score: {score:.2f}\n"
                f"     Preview: {content_preview}..."
            )
        knowledge_text = "\n".join(knowledge_items)

    # Format Web Search Resolution
    web_search_text = "No web search performed."
    if web_resolution.get("resolution_steps"):
        steps = web_resolution.get("resolution_steps", [])
        sources = web_resolution.get("sources", [])
        confidence = web_resolution.get("confidence", 0.0)
        query_used = web_resolution.get("search_query_used", "")

        steps_text = "\n".join([f"    {i+1}. {step}" for i, step in enumerate(steps)])
        sources_text = "\n".join([f"    - {source}" for source in sources[:5]])

        web_search_text = f"""Query Used: {query_used}
  Confidence: {confidence:.2f}

  Resolution Steps:
{steps_text}

  External Sources:
{sources_text}"""

    # Format Knowledge Sufficiency
    sufficiency_text = "Not evaluated."
    if knowledge_sufficiency:
        is_sufficient = knowledge_sufficiency.get("is_sufficient", False)
        best_score = knowledge_sufficiency.get("best_score", 0.0)
        best_source = knowledge_sufficiency.get("best_source", "none")
        sufficiency_text = f"""Sufficient: {"Yes" if is_sufficient else "No"}
  Best Score: {best_score:.2f}
  Best Source: {best_source}"""

    # Format Augmentation Metadata
    augmentation_text = "No augmentation performed."
    if augmentation_source and augmentation_source != "none":
        augmentation_text = f"""Source: {augmentation_source}
  Iterations: {augmentation_iteration}
  Note: Context Grounding results above include augmented searches."""

    # Format collected information from all sources
    collected_info = f"""[1] CONFLUENCE MEMORY (Previous Similar Tickets):
{memory_text}

[2] CONTEXT GROUNDING (Internal Knowledge Base):
{knowledge_text}

[3] WEB SEARCH RESOLUTION (External Sources):
{web_search_text}

[4] KNOWLEDGE SUFFICIENCY EVALUATION:
{sufficiency_text}

[5] AUGMENTATION METADATA:
{augmentation_text}"""

    # Build prompt using centralized content - requesting JSON output
    json_format_instruction = f"""Return a valid JSON object matching this structure:
{{
  "issue_summary": "1-2 sentences summarizing the issue",
  "initial_assessment": {{
    "issue_type": "Type of issue",
    "urgency_level": "High/Medium/Low",
    "complexity_estimate": "Simple/Moderate/Complex",
    "information_quality": "Complete/Partial/Limited"
  }},
  "key_findings": ["insight 1", "insight 2", "insight 3"],
  "investigation_steps": ["step 1", "step 2", "step 3", ...],
  "knowledge_base_references": ["reference 1", "reference 2", ...],
  "estimated_resolution_time": "time estimate with justification",
  "recommended_next_actions": ["action 1", "action 2", ...]
}}"""

    prompt = (PromptBuilder("GENERATE INVESTIGATION ACTIONS")
        .add_section(None, CATO_AGENT_GUIDELINES)
        .add_section("YOUR TASK", INVESTIGATION_TASK)
        .add_section("TICKET CONTEXT", ticket_context)
        .add_section("COLLECTED INFORMATION FROM ALL SOURCES", collected_info)
        .add_section("YOUR ANALYSIS TASK", INVESTIGATION_ANALYSIS_TASK)
        .add_section("CONTENT GUIDELINES", INVESTIGATION_FORMAT)
        .add_section("JSON RESPONSE FORMAT (CRITICAL)", json_format_instruction)
        .add_section("YOUR RESPONSE", "Generate the investigation plan as valid JSON now:")
        .set_response_format("json")
        .build())

    logger.debug("Sending investigation steps generation to LLM (JSON response expected)")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=1200,
        temperature=0.2,
        operation_name="generate_investigation_steps"
    )

    log_llm_response(logger, response, "INVESTIGATION STEPS (JSON)")

    # Parse JSON response into InvestigationResponse model
    import json
    try:
        response_json = json.loads(response.strip())
        investigation_response = InvestigationResponse(**response_json)
        logger.info(f"[Step 6] Successfully parsed investigation response into structured object")
        return investigation_response
    except json.JSONDecodeError as e:
        logger.error(f"[Step 6] Failed to parse JSON response: {e}")
        raise LLMError(
            "LLM returned invalid JSON for investigation response",
            details={"response": response[:500], "error": str(e)}
        )
    except Exception as e:
        logger.error(f"[Step 6] Failed to validate investigation response: {e}")
        raise LLMError(
            "LLM response does not match InvestigationResponse schema",
            details={"response": response[:500], "error": str(e)}
        )
