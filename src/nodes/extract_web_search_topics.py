"""
Step 5d: Extract Web Search Topics

Analyzes web search results and extracts specific topics to search in
internal knowledge base (Context Grounding + Confluence Memory).

Strategy:
1. LLM analyzes web search resolution steps and sources
2. Identifies application names, procedures, and Cato-specific topics
3. Generates 2-3 targeted queries optimized for internal KB search
4. Adds "Cato" context where relevant for org-specific documentation

Example:
  Web: "Slack Desktop requires VPN connection"
  → Topics: ["Slack Desktop", "Cato VPN", "VPN requirements"]
  → Queries: ["Slack Desktop Cato VPN configuration", "Slack VPN requirements"]
"""

import logging
from src.graph.state import GraphState
from src.utils.llm_service import get_llm_service
from src.utils.prompt_utils import log_llm_response, parse_llm_json_response
from src.config.prompts import CATO_AGENT_GUIDELINES, JSON_INSTRUCTIONS

logger = logging.getLogger(__name__)


async def extract_web_search_topics(state: GraphState) -> dict:
    """
    Extract topics from web search results for targeted KB searches.

    Uses LLM to analyze web search findings and identify specific topics
    that should be searched in internal knowledge base.

    Args:
        state: Current graph state containing:
            - web_search_resolution: Output from web_search_node

    Returns:
        dict: {
            "web_search_topics": {
                "has_topics": bool,
                "topics": List[str],
                "targeted_kb_queries": List[str],
                "reasoning": str
            }
        }
    """
    web_resolution = state.get("web_search_resolution", {})

    # Check if web search returned results
    if not web_resolution.get("resolution_steps"):
        logger.info("[Web Topics] No web search results to analyze")
        return {
            "web_search_topics": {
                "has_topics": False,
                "topics": [],
                "targeted_kb_queries": [],
                "reasoning": "No web search results available"
            }
        }

    logger.info("[Web Topics] Analyzing web search results for internal KB queries")

    try:
        # Format web search results for LLM analysis
        steps_text = "\n".join([
            f"  {i+1}. {step}"
            for i, step in enumerate(web_resolution.get("resolution_steps", []))
        ])

        sources_text = "\n".join([
            f"  - {source}"
            for source in web_resolution.get("sources", [])
        ])

        query_used = web_resolution.get("search_query_used", "")
        confidence = web_resolution.get("confidence", 0.0)

        logger.info(f"[Web Topics] Web search confidence: {confidence:.2f}")
        logger.info(f"[Web Topics] Steps to analyze: {len(web_resolution.get('resolution_steps', []))}")

        llm_service = get_llm_service()

        prompt = f"""{CATO_AGENT_GUIDELINES}

═══════════════════════════════════════════════════════════════════════════════
TASK: EXTRACT TOPICS FROM WEB SEARCH FOR INTERNAL KB SEARCH
═══════════════════════════════════════════════════════════════════════════════

You are analyzing external web search results to identify specific topics we should
search in our INTERNAL knowledge base (Context Grounding + Confluence).

WEB SEARCH RESULTS:

Query Used: {query_used}
Confidence: {confidence:.2f}

Resolution Steps:
{steps_text}

Sources:
{sources_text}

═══════════════════════════════════════════════════════════════════════════════
YOUR ANALYSIS TASK
═══════════════════════════════════════════════════════════════════════════════

Identify specific topics we should search in INTERNAL Cato knowledge base:

1. **Application/System Names**
   - What specific applications are mentioned? (e.g., "Slack Desktop", "JumpCloud", "Microsoft 365")
   - What systems or platforms? (e.g., "Windows 10", "iPhone", "VPN client")

2. **Technical Procedures**
   - What procedures are described? (e.g., "password reset", "MFA setup", "SSO configuration")
   - What actions are required? (e.g., "VPN connection", "admin approval")

3. **Cato-Specific Context**
   - What likely has Cato-specific documentation?
   - VPN/network topics → add "Cato VPN"
   - SSO/authentication → add "Cato SSO"
   - Client applications → add "Cato Client"

4. **Prerequisites/Dependencies**
   - What requirements are mentioned? (e.g., "VPN required", "admin rights needed")

═══════════════════════════════════════════════════════════════════════════════
QUERY GENERATION RULES
═══════════════════════════════════════════════════════════════════════════════

Generate 2-3 targeted queries for internal KB search:

✓ DO:
  - Add "Cato" for VPN/network/SSO topics
  - Make procedural: "reset procedure" not just "reset"
  - Combine related terms: "Slack Desktop Cato VPN configuration"
  - Be specific: "JumpCloud SSO setup" not "authentication"

✗ DON'T:
  - Generic queries: "email settings", "password help"
  - Duplicate what was already searched initially
  - More than 3 queries

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

Example 1 - Slack VPN Issue:
Web mentions: "Slack Desktop requires VPN connection"
{{
  "has_topics": true,
  "topics": ["Slack Desktop", "Cato VPN", "VPN requirements"],
  "targeted_kb_queries": [
    "Slack Desktop Cato VPN configuration",
    "Slack VPN connection requirements Cato"
  ],
  "reasoning": "Web indicates Slack needs VPN. Should check for Cato-specific Slack VPN setup docs."
}}

Example 2 - SSO Password Reset:
Web mentions: "Reset password via SSO portal"
{{
  "has_topics": true,
  "topics": ["SSO portal", "password reset procedure", "self-service"],
  "targeted_kb_queries": [
    "Cato SSO password reset procedure",
    "self-service password reset SSO"
  ],
  "reasoning": "Web describes SSO password reset. Should check for Cato SSO-specific procedure."
}}

Example 3 - Generic Troubleshooting (No Cato-Specific Topics):
Web mentions: "Check email settings in Outlook"
{{
  "has_topics": false,
  "topics": [],
  "targeted_kb_queries": [],
  "reasoning": "Generic Outlook troubleshooting, unlikely to have org-specific documentation."
}}

═══════════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════════════════

{JSON_INSTRUCTIONS}

Return ONLY valid JSON:

{{
  "has_topics": true or false,
  "topics": ["specific topic 1", "specific topic 2"],
  "targeted_kb_queries": ["specific query 1", "specific query 2"],
  "reasoning": "Clear explanation of what topics were identified and why they're relevant for internal KB"
}}

IMPORTANT:
- Set has_topics=false if no Cato-specific topics found
- Max 3 queries
- Each query should be 3-7 words
- Be specific, not generic

JSON Response:"""

        logger.debug("[Web Topics] Sending to LLM for topic extraction")

        response = await llm_service.invoke(
            prompt=prompt,
            max_tokens=400,
            temperature=0.2  # Low temperature for consistent extraction
        )

        log_llm_response(logger, response, "WEB TOPICS EXTRACTION")

        result = parse_llm_json_response(response, logger, "web search topics")

        # Validate structure
        has_topics = result.get("has_topics", False)
        topics = result.get("topics", [])
        queries = result.get("targeted_kb_queries", [])

        logger.info(f"[Web Topics] Has topics: {has_topics}")

        if has_topics:
            logger.info(f"[Web Topics] Extracted {len(topics)} topics: {topics}")
            logger.info(f"[Web Topics] Generated {len(queries)} KB queries:")
            for i, query in enumerate(queries, 1):
                logger.info(f"[Web Topics]   {i}. {query}")
        else:
            logger.info("[Web Topics] No relevant topics found for internal KB search")

        return {
            "web_search_topics": result
        }

    except Exception as e:
        logger.error(f"[Web Topics] Failed to extract topics: {str(e)}", exc_info=True)
        logger.warning("[Web Topics] Returning has_topics=False due to error")

        # Return safe default on error
        return {
            "web_search_topics": {
                "has_topics": False,
                "topics": [],
                "targeted_kb_queries": [],
                "reasoning": f"Failed to extract topics due to error: {str(e)}"
            }
        }
