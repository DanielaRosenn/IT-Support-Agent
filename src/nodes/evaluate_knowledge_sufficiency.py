"""
Step 5b: Evaluate Knowledge Sufficiency

Determines if collected knowledge (memory + context grounding) is sufficient
based on configurable threshold (default 0.8).

If score > threshold: Knowledge is sufficient, skip web search
If score ≤ threshold: Knowledge is insufficient, trigger web search
"""

import logging
from src.graph.state import GraphState
from src.config import config
from src.config.assets_loader import get_knowledge_threshold

logger = logging.getLogger(__name__)


async def evaluate_knowledge_sufficiency(state: GraphState) -> dict:
    """
    Evaluate if knowledge sources are sufficient.

    Checks memory_results, freshservice_articles, and context_grounding_results
    for the highest relevance score. Compares against KNOWLEDGE_SUFFICIENCY_THRESHOLD
    from config.

    Args:
        state: Current graph state containing:
            - memory_results: List of Confluence memory search results
            - freshservice_articles: List of FreshService articles with semantic scores
            - context_grounding_results: List of Context Grounding results

    Returns:
        dict: {
            "knowledge_sufficiency": {
                "is_sufficient": bool,
                "best_score": float,
                "best_source": str
            }
        }
    """
    memory_results = state.get("memory_results", [])
    freshservice_articles = state.get("freshservice_articles", [])
    cg_results = state.get("context_grounding_results", [])

    logger.info(f"[Knowledge Sufficiency] Evaluating {len(memory_results)} memory + {len(freshservice_articles)} articles + {len(cg_results)} CG results")

    best_score = 0.0
    best_source = "none"

    # Check memory results for best score
    for result in memory_results:
        score = result.get("score", 0.0)
        if score > best_score:
            best_score = score
            best_source = f"Memory: {result.get('source', 'unknown')}"

    # Check FreshService articles for best score (these have semantic re-ranking scores)
    # FreshService articles are the most reliable - they're specifically relevant
    # Note: FreshService articles are dicts (from UiPath), safe to use .get()
    for article in freshservice_articles:
        # FreshService articles have relevance_score from re-ranking (not semantic_score)
        score = article.get("relevance_score", article.get("semantic_score", 0.0))
        if score > best_score:
            best_score = score
            best_source = f"FreshService: {article.get('title', 'unknown')}"
            logger.debug(f"[Knowledge Sufficiency] New best from FreshService: {best_score:.2f}")

    # Check context grounding results for best score
    # Note: CG results are Pydantic models, use getattr()
    for result in cg_results:
        # Handle both dict and Pydantic model formats
        if hasattr(result, 'score'):
            score = getattr(result, 'score', 0.0)
            source = getattr(result, 'source', 'unknown')
        else:
            score = result.get("score", 0.0)
            source = result.get('source', 'unknown')

        if score > best_score:
            best_score = score
            best_source = f"Context Grounding: {source}"
            logger.debug(f"[Knowledge Sufficiency] New best from CG: {best_score:.2f}")

    # Determine sufficiency: If ANY source has good score, we're sufficient
    # FreshService articles with 0.8+ are very reliable
    # Threshold loaded from UiPath Assets (falls back to config.py if not found)
    threshold = get_knowledge_threshold()
    is_sufficient = best_score > threshold

    logger.info(f"[Knowledge Sufficiency] Best score: {best_score:.2f} from {best_source}")
    logger.info(f"[Knowledge Sufficiency] Threshold: {threshold} (from UiPath Assets or config.py)")
    logger.info(f"[Knowledge Sufficiency] Is sufficient: {is_sufficient}")

    if is_sufficient:
        logger.info(f"[Knowledge Sufficiency] ✓ Knowledge is sufficient (>{threshold}), will skip web search")
    else:
        logger.info(f"[Knowledge Sufficiency] ✗ Knowledge is insufficient (≤{threshold}), will trigger web search")

    return {
        "knowledge_sufficiency": {
            "is_sufficient": is_sufficient,
            "best_score": best_score,
            "best_source": best_source
        }
    }
