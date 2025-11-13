"""
Step 2: Search Confluence Memory

Searches for similar previously resolved tickets using LLM-based query refinement.
"""

import logging
from typing import Dict, Any, List
from src.graph.state import GraphState
from src.integrations.confluence_memory import ConfluenceMemoryIntegration
from src.utils.keyword_extraction import extract_keywords_for_confluence_memory
from src.config import config

logger = logging.getLogger(__name__)


async def search_memory(state: GraphState) -> dict:
    """
    Step 2: Search Confluence Memory for similar resolved tickets.

    Uses LLM to extract key issue terms first, then searches with refined keywords.

    Args:
        state: Current graph state containing ticket_info

    Returns:
        dict: Updated state with memory_results populated
    """
    ticket_info = state["ticket_info"]

    logger.info(f"[Step 2] Searching Confluence Memory for similar tickets")
    logger.info(f"  - Category: {ticket_info.category}")
    logger.info(f"  - Subject: {ticket_info.subject}")

    try:
        # [Step 2a] Use LLM to extract key search terms
        logger.info("[Step 2a] Extracting key search terms using LLM...")

        search_query = await extract_keywords_for_confluence_memory(
            category=ticket_info.category,
            subject=ticket_info.subject,
            description=ticket_info.description
        )

        logger.info(f"[Step 2a] Refined search query: '{search_query}'")

        # [Step 2b] Search Confluence Memory with refined query
        logger.info("[Step 2b] Searching Confluence Memory...")

        memory_integration = ConfluenceMemoryIntegration()
        results = memory_integration.search_similar_resolutions(
            query=search_query,
            number_of_results=3  # Get top 3 matches
        )

        # [Step 2c] Filter results by minimum score threshold
        min_score_threshold = config.CONFLUENCE_MEMORY_MIN_SCORE
        high_score_threshold = config.CONFLUENCE_MEMORY_HIGH_SCORE

        # Filter out low-scoring results
        filtered_results = [r for r in results if r.score >= min_score_threshold]

        logger.info(f"[Step 2c] Filtering memory results:")
        logger.info(f"  - Total results: {len(results)}")
        logger.info(f"  - After filtering (≥{min_score_threshold}): {len(filtered_results)}")

        # Identify high-scoring matches for logging
        high_score_matches = [r for r in filtered_results if r.score >= high_score_threshold]

        if high_score_matches:
            logger.info(f"[Step 2] Found {len(high_score_matches)} high-scoring matches (≥{high_score_threshold}):")
            for match in high_score_matches:
                logger.info(f"  - Score: {match.score:.3f} | Source: {match.source}")
        else:
            logger.info(f"[Step 2] No high-scoring matches found (threshold: {high_score_threshold})")
            if filtered_results:
                logger.info(f"  - Best match score: {filtered_results[0].score:.3f}")

        # Convert filtered results to dicts for state storage
        memory_results = [
            {
                "content": r.content,
                "source": r.source,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in filtered_results
        ]

        logger.info(f"[Step 2] Successfully retrieved {len(memory_results)} memory results")

        # Log all source pages found
        if memory_results:
            logger.info(f"[Step 2] Source pages:")
            for i, result in enumerate(memory_results, 1):
                logger.info(f"  [{i}] Score: {result['score']:.3f} - {result['source']}")

        return {
            "memory_results": memory_results
        }

    except Exception as e:
        logger.error(f"[Step 2] Failed to search Confluence Memory: {str(e)}")
        # Don't fail the entire flow - just return empty results
        logger.warning("[Step 2] Continuing with empty memory results")
        return {
            "memory_results": []
        }