"""
Step 5c: Web Search Node

Wrapper node that integrates WebSearchResolutionWorkflow into LangGraph.
Performs web search when knowledge base results are insufficient (score ≤ 0.8).

This node:
1. Extracts ticket info from state
2. Calls WebSearchResolutionWorkflow (with DuckDuckGo search + Claude ranking)
3. Returns structured results back to state

All settings are configurable via config.py
"""

import logging
from src.graph.state import GraphState
from src.workflows.web_search_resolution import WebSearchResolutionWorkflow
from src.config import config

logger = logging.getLogger(__name__)


async def web_search_node(state: GraphState) -> dict:
    """
    Perform web search using WebSearchResolutionWorkflow.

    Wrapper that adapts the standalone WebSearchResolutionWorkflow to work
    within the LangGraph state machine.

    Args:
        state: Current graph state containing:
            - ticket_info: TicketInfo object with ticket details

    Returns:
        dict: {
            "web_search_resolution": {
                "resolution_steps": List[str],
                "sources": List[str],
                "search_query_used": str,
                "confidence": float
            }
        }

    Configuration (from config.py):
        - WEB_SEARCH_ENABLED: Enable/disable web search
        - WEB_SEARCH_MAX_RESULTS: Max results to fetch
        - WEB_SEARCH_TRUSTED_DOMAINS: Whitelist of allowed domains
        - WEB_SEARCH_TOP_RESULTS_TO_USE: Top N results after ranking
        - WEB_SEARCH_MIN_CONFIDENCE: Minimum confidence threshold
    """
    ticket_info = state["ticket_info"]

    logger.info(f"[Web Search] Starting for ticket {ticket_info.ticket_id}")
    logger.info(f"[Web Search] Subject: {ticket_info.subject}")

    # Check if web search is enabled in config
    if not config.WEB_SEARCH_ENABLED:
        logger.warning("[Web Search] Web search is disabled in config, skipping")
        return {
            "web_search_resolution": {
                "resolution_steps": [],
                "sources": [],
                "search_query_used": "",
                "confidence": 0.0
            }
        }

    try:
        # Initialize WebSearchResolutionWorkflow
        # The workflow uses config settings internally for:
        # - Max results (WEB_SEARCH_MAX_RESULTS)
        # - Trusted domains (WEB_SEARCH_TRUSTED_DOMAINS)
        # - Timeout (WEB_SEARCH_TIMEOUT_SECONDS)
        # - Top results to use (WEB_SEARCH_TOP_RESULTS_TO_USE)
        workflow = WebSearchResolutionWorkflow()

        logger.info(f"[Web Search] Calling WebSearchResolutionWorkflow")
        logger.info(f"[Web Search] Max results: {config.WEB_SEARCH_MAX_RESULTS}")
        logger.info(f"[Web Search] Top results to use: {config.WEB_SEARCH_TOP_RESULTS_TO_USE}")
        logger.info(f"[Web Search] Trusted domains: {len(config.WEB_SEARCH_TRUSTED_DOMAINS)}")

        # Call the workflow (it handles all 4 steps internally)
        resolution = await workflow.generate_resolution(
            ticket=ticket_info,
            max_search_results=config.WEB_SEARCH_MAX_RESULTS
        )

        logger.info(f"[Web Search] ✓ Completed successfully")
        logger.info(f"[Web Search] Confidence: {resolution.confidence:.2f}")
        logger.info(f"[Web Search] Steps generated: {len(resolution.resolution_steps)}")
        logger.info(f"[Web Search] Sources: {len(resolution.sources)}")
        logger.info(f"[Web Search] Query used: {resolution.search_query_used}")

        # Check if confidence meets minimum threshold
        if resolution.confidence < config.WEB_SEARCH_MIN_CONFIDENCE:
            logger.warning(
                f"[Web Search] Low confidence: {resolution.confidence:.2f} "
                f"< {config.WEB_SEARCH_MIN_CONFIDENCE} (min threshold)"
            )

        # Convert WebSearchResolution to dict for state
        return {
            "web_search_resolution": {
                "resolution_steps": resolution.resolution_steps,
                "sources": resolution.sources,
                "search_query_used": resolution.search_query_used,
                "confidence": resolution.confidence
            }
        }

    except Exception as e:
        logger.error(f"[Web Search] Failed: {str(e)}", exc_info=True)
        logger.warning("[Web Search] Returning empty results due to error")

        # Return empty results on failure (graceful degradation)
        return {
            "web_search_resolution": {
                "resolution_steps": [],
                "sources": [],
                "search_query_used": "",
                "confidence": 0.0
            }
        }
