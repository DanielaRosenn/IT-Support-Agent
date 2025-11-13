"""
Step 6a: Check Missing Information

Analyzes all collected knowledge to determine if sufficient information exists
for generating a high-quality response, or if targeted augmentation is needed.

Strategy:
1. LLM analyzes ALL sources (memory, context grounding, web search, existing augmentation)
2. Identifies specific gaps or missing details
3. Generates 2-3 targeted queries for knowledge augmentation
4. Respects max iteration limit (2) to prevent infinite loops

Example Flow:
  Sources: Generic VPN setup, no Slack-specific info
  → Missing: "Slack Desktop Cato VPN configuration"
  → Query: "Slack Desktop VPN requirements Cato"
"""

import logging
from typing import List, Dict, Any
from src.graph.state import GraphState
from src.utils.llm_service import get_llm_service
from src.utils.prompt_utils import (
    log_llm_response,
    parse_llm_json_response,
    build_ticket_context,
    PromptBuilder
)
from src.config.prompts import CATO_AGENT_GUIDELINES, JSON_INSTRUCTIONS
from src.config import config

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _format_source_list(results: List[Dict], source_name: str, max_results: int = 5) -> str:
    """
    Format a list of search results (memory, context grounding) into text.

    Args:
        results: List of result dicts with 'content' and 'score'
        source_name: Name for logging (e.g., "Memory", "Context Grounding")
        max_results: Maximum number of results to include

    Returns:
        Formatted string or "None" if no results
    """
    if not results:
        return "None"

    items = []
    for i, result in enumerate(results[:max_results], 1):
        # Handle both Pydantic models and dicts
        if hasattr(result, 'content'):
            content_preview = getattr(result, 'content', '')[:200]
            score = getattr(result, 'score', 0.0)
        else:
            content_preview = result.get('content', '')[:200]
            score = result.get('score', 0.0)

        items.append(f"  {i}. Score: {score:.2f} | {content_preview}...")

    return "\n".join(items)


def _format_all_sources(
    memory_results: List[Dict],
    cg_results: List[Dict],
    web_resolution: Dict,
    web_topics: Dict
) -> str:
    """
    Format all information sources into a single text summary.

    Args:
        memory_results: Confluence memory search results
        cg_results: Context grounding search results
        web_resolution: Web search resolution dict
        web_topics: Extracted web topics dict

    Returns:
        Formatted multi-section text summary
    """
    sections = []

    # Memory results
    memory_text = _format_source_list(memory_results, "Memory")
    sections.append(f"[1] CONFLUENCE MEMORY RESULTS:\n{memory_text}")

    # Context grounding results
    cg_text = _format_source_list(cg_results, "Context Grounding")
    sections.append(f"[2] CONTEXT GROUNDING RESULTS:\n{cg_text}")

    # Web search resolution
    web_text = "None"
    if web_resolution.get("resolution_steps"):
        steps = web_resolution.get("resolution_steps", [])
        web_text = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(steps)])
    sections.append(f"[3] WEB SEARCH RESOLUTION:\n{web_text}")

    # Web topics
    web_topics_text = "None"
    if web_topics.get("has_topics"):
        topics = web_topics.get("topics", [])
        queries = web_topics.get("targeted_kb_queries", [])
        web_topics_text = f"Topics: {', '.join(topics)}\nQueries: {', '.join(queries)}"
    sections.append(f"[4] WEB SEARCH TOPICS (if extracted):\n{web_topics_text}")

    return "\n\n".join(sections)


def _build_gap_analysis_instructions(current_iteration: int, max_iterations: int) -> str:
    """
    Build the analysis instructions for the gap analysis prompt.

    Args:
        current_iteration: Current augmentation iteration
        max_iterations: Maximum allowed iterations

    Returns:
        Formatted instructions text
    """
    return f"""Evaluate if the collected information is SUFFICIENT to generate a complete,
actionable response that addresses the ticket comprehensively.

**COMPLETENESS CRITERIA:**

Ask yourself:
1. **Application-Specific Details**
   - Do we have Cato-specific configuration for mentioned applications?
   - Are vendor-specific procedures covered?

2. **Procedural Completeness**
   - Are all steps clearly outlined?
   - Are prerequisites and dependencies mentioned?
   - Is there information about required permissions/access?

3. **Context & Integration**
   - Do we understand how this integrates with Cato systems?
   - Are VPN/network/SSO dependencies addressed?
   - Is there org-specific context?

4. **Troubleshooting Details**
   - Are error scenarios covered?
   - Is there fallback guidance?
   - Are escalation paths mentioned?

═══════════════════════════════════════════════════════════════════════════════
WHEN TO AUGMENT (needs_augmentation=true)
═══════════════════════════════════════════════════════════════════════════════

Set needs_augmentation=true if:
✓ Application mentioned but no Cato-specific config found
✓ Procedure described generically, missing org-specific steps
✓ Missing prerequisites (VPN, SSO, admin access requirements)
✓ Web search found solution but no internal validation/customization
✓ High-level guidance exists but lacks actionable details

═══════════════════════════════════════════════════════════════════════════════
WHEN NOT TO AUGMENT (needs_augmentation=false)
═══════════════════════════════════════════════════════════════════════════════

Set needs_augmentation=false if:
✗ Comprehensive procedure with Cato-specific steps exists
✗ Multiple sources confirm the same approach
✗ Actionable, detailed instructions are available
✗ Already at max iterations ({max_iterations})
✗ Ticket is too vague/generic for targeted search

═══════════════════════════════════════════════════════════════════════════════
QUERY GENERATION RULES (if augmentation needed)
═══════════════════════════════════════════════════════════════════════════════

Generate 2-3 targeted queries that fill SPECIFIC gaps:

✓ DO:
  - Focus on missing application/system details
  - Add "Cato" for org-specific context
  - Make procedural: "setup procedure" not "setup"
  - Target specific integrations: "Slack Cato VPN configuration"

✗ DON'T:
  - Repeat queries already tried
  - Be too generic: "email settings"
  - Generate more than 3 queries
  - Query for information already found

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

Example 1 - Application-Specific Gap:
Sources: Generic "reset password via SSO", no Cato SSO details
{{
  "needs_augmentation": true,
  "missing_topics": ["Cato SSO password reset procedure", "self-service portal"],
  "targeted_queries": [
    "Cato SSO password reset procedure",
    "self-service password reset portal Cato"
  ],
  "reasoning": "Web provides generic SSO guidance but no Cato-specific procedure. Need internal documentation for self-service portal and org-specific steps."
}}

Example 2 - Integration Gap:
Sources: "Slack requires VPN", web search confirms, but no Cato VPN config
{{
  "needs_augmentation": true,
  "missing_topics": ["Slack Desktop Cato VPN configuration", "Slack VPN prerequisites"],
  "targeted_queries": [
    "Slack Desktop Cato VPN configuration",
    "Slack VPN connection requirements"
  ],
  "reasoning": "Web confirms Slack needs VPN but no internal docs on Cato VPN configuration for Slack. Need org-specific setup instructions."
}}

Example 3 - Complete Information:
Sources: Confluence memory with detailed procedure, CG confirms same approach
{{
  "needs_augmentation": false,
  "missing_topics": [],
  "targeted_queries": [],
  "reasoning": "Comprehensive procedure found in Confluence memory with step-by-step instructions. Context grounding confirms same approach. Sufficient to generate response."
}}

Example 4 - Max Iterations Reached:
Current iteration: {max_iterations}
{{
  "needs_augmentation": false,
  "missing_topics": [],
  "targeted_queries": [],
  "reasoning": "Max augmentation iterations reached. Proceeding with available information."
}}

Return ONLY valid JSON:

{{
  "needs_augmentation": true or false,
  "missing_topics": ["specific missing topic 1", "specific missing topic 2"],
  "targeted_queries": ["specific query 1", "specific query 2"],
  "reasoning": "Clear explanation of what's missing and why augmentation is/isn't needed"
}}

IMPORTANT:
- Be specific about gaps (not generic)
- Max 3 queries
- Each query should be 3-7 words
- Set needs_augmentation=false if information is sufficient OR max iterations reached

Current Augmentation Iteration: {current_iteration}/{max_iterations}"""


async def _analyze_information_gaps(
    ticket_info: Any,
    sources_text: str,
    current_iteration: int,
    max_iterations: int
) -> Dict[str, Any]:
    """
    Use LLM to analyze information completeness and identify gaps.

    Args:
        ticket_info: TicketInfo object
        sources_text: Formatted text of all information sources
        current_iteration: Current augmentation iteration
        max_iterations: Maximum allowed iterations

    Returns:
        Dict with needs_augmentation, missing_topics, targeted_queries, reasoning

    Raises:
        LLMError: If LLM invocation or parsing fails
    """
    llm_service = get_llm_service()

    # Build ticket context
    ticket_context = build_ticket_context(ticket_info)

    # Build analysis instructions
    analysis_instructions = _build_gap_analysis_instructions(current_iteration, max_iterations)

    # Build prompt using PromptBuilder
    prompt = (PromptBuilder("ANALYZE INFORMATION COMPLETENESS & IDENTIFY GAPS")
        .add_section("TICKET INFORMATION", ticket_context)
        .add_section("COLLECTED INFORMATION FROM ALL SOURCES", sources_text)
        .add_section("YOUR ANALYSIS TASK", analysis_instructions)
        .build())

    logger.debug("[Missing Info] Sending completeness analysis to LLM")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=500,
        temperature=0.3  # Slightly higher for nuanced gap analysis
    )

    # Log and parse response
    log_llm_response(logger, response, "MISSING INFORMATION CHECK")
    result = parse_llm_json_response(response, logger, "missing information check")

    return result


def _process_gap_analysis_result(
    result: Dict[str, Any],
    current_iteration: int
) -> Dict[str, Any]:
    """
    Process gap analysis result and build state update.

    Args:
        result: LLM analysis result dict
        current_iteration: Current augmentation iteration

    Returns:
        State update dict with missing_information_check and optionally augmentation_iteration
    """
    needs_augmentation = result.get("needs_augmentation", False)
    missing_topics = result.get("missing_topics", [])
    targeted_queries = result.get("targeted_queries", [])

    logger.info(f"[Missing Info] Needs augmentation: {needs_augmentation}")

    if needs_augmentation:
        logger.info(f"[Missing Info] Missing topics: {missing_topics}")
        logger.info(f"[Missing Info] Generated {len(targeted_queries)} targeted queries:")
        for i, query in enumerate(targeted_queries, 1):
            logger.info(f"[Missing Info]   {i}. {query}")

        # Increment iteration counter for next pass
        new_iteration = current_iteration + 1
        logger.info(f"[Missing Info] Incrementing iteration: {current_iteration} → {new_iteration}")

        return {
            "missing_information_check": result,
            "augmentation_iteration": new_iteration
        }
    else:
        logger.info("[Missing Info] Information is complete, no augmentation needed")
        logger.info(f"[Missing Info] Reasoning: {result.get('reasoning', 'N/A')}")

        return {
            "missing_information_check": result
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================


async def check_missing_information(state: GraphState) -> dict:
    """
    Analyze collected knowledge for gaps and determine if augmentation is needed.

    Main orchestrator function that:
    1. Checks iteration limit
    2. Formats all information sources
    3. Analyzes gaps using LLM
    4. Processes and returns results

    Args:
        state: Current graph state containing:
            - ticket_info: Original ticket details
            - memory_results: Confluence memory results (if any)
            - context_grounding_results: Context grounding results (if any)
            - web_search_resolution: Web search results (if any)
            - web_search_topics: Extracted web topics (if any)
            - augmentation_iteration: Current iteration count (default 0)

    Returns:
        dict: {
            "missing_information_check": {
                "needs_augmentation": bool,
                "missing_topics": List[str],
                "targeted_queries": List[str],
                "reasoning": str
            },
            "augmentation_iteration": int (incremented if needed)
        }
    """
    # Extract state variables
    ticket_info = state["ticket_info"]
    memory_results = state.get("memory_results", [])
    cg_results = state.get("context_grounding_results", [])
    web_resolution = state.get("web_search_resolution", {})
    web_topics = state.get("web_search_topics", {})
    current_iteration = state.get("augmentation_iteration", 0)

    logger.info(f"[Missing Info] Checking completeness (iteration {current_iteration}/{config.MAX_AUGMENTATION_ITERATIONS})")

    # [Step 1] Check iteration limit (early return)
    if current_iteration >= config.MAX_AUGMENTATION_ITERATIONS:
        logger.warning(f"[Missing Info] Max iterations ({config.MAX_AUGMENTATION_ITERATIONS}) reached, stopping augmentation")
        return {
            "missing_information_check": {
                "needs_augmentation": False,
                "missing_topics": [],
                "targeted_queries": [],
                "reasoning": f"Max augmentation iterations ({config.MAX_AUGMENTATION_ITERATIONS}) reached"
            }
        }

    # [Step 2] Format all collected information
    logger.info(f"[Missing Info] Analyzing {len(memory_results)} memory + {len(cg_results)} CG + web search results")

    try:
        sources_text = _format_all_sources(
            memory_results=memory_results,
            cg_results=cg_results,
            web_resolution=web_resolution,
            web_topics=web_topics
        )

        # [Step 3] Analyze gaps using LLM
        result = await _analyze_information_gaps(
            ticket_info=ticket_info,
            sources_text=sources_text,
            current_iteration=current_iteration,
            max_iterations=config.MAX_AUGMENTATION_ITERATIONS
        )

        # [Step 4] Process result and return state update
        return _process_gap_analysis_result(result, current_iteration)

    except Exception as e:
        # Error handling: return safe default
        logger.error(f"[Missing Info] Failed to check completeness: {str(e)}", exc_info=True)
        logger.warning("[Missing Info] Returning needs_augmentation=False due to error")

        return {
            "missing_information_check": {
                "needs_augmentation": False,
                "missing_topics": [],
                "targeted_queries": [],
                "reasoning": f"Failed to analyze completeness due to error: {str(e)}"
            }
        }
