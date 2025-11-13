"""
Step 6b: Augment Knowledge

Executes targeted knowledge base searches based on identified gaps from either:
1. Web search topic extraction (web_search_topics)
2. Missing information check (missing_information_check)

Strategy:
1. Determine augmentation source (web topics or missing info check)
2. Execute targeted searches in Context Grounding
3. Append new results to existing context_grounding_results
4. Track augmentation source for transparency

Example Flow:
  Input: ["Slack Desktop Cato VPN configuration", "Slack VPN requirements"]
  → Search Context Grounding with each query
  → Append high-quality results (score > 0.5) to context_grounding_results
  → Update state with augmentation_source="web_topics"
"""

import logging
from src.graph.state import GraphState
from src.integrations.uipath_context_grounding import search_knowledge_base
from src.config import config

logger = logging.getLogger(__name__)


async def augment_knowledge(state: GraphState) -> dict:
    """
    Perform targeted knowledge base searches to fill identified gaps.

    Executes searches from either web_search_topics or missing_information_check,
    appends results to existing context_grounding_results.

    Args:
        state: Current graph state containing:
            - context_grounding_results: Existing CG results (to append to)
            - web_search_topics: Optional web-guided queries
            - missing_information_check: Optional gap-filling queries
            - augmentation_iteration: Current iteration count

    Returns:
        dict: {
            "context_grounding_results": List[Dict] (appended with new results),
            "augmentation_source": str ("web_topics" or "missing_info")
        }
    """
    existing_cg_results = state.get("context_grounding_results", [])
    web_topics = state.get("web_search_topics", {})
    missing_info_check = state.get("missing_information_check", {})
    current_iteration = state.get("augmentation_iteration", 0)

    logger.info(f"[Augment] Starting knowledge augmentation (iteration {current_iteration})")
    logger.info(f"[Augment] Existing CG results: {len(existing_cg_results)}")

    # Determine augmentation source and queries
    targeted_queries = []
    augmentation_source = None

    # Priority 1: Web search topics (more specific, directly from web analysis)
    if web_topics.get("has_topics") and web_topics.get("targeted_kb_queries"):
        targeted_queries = web_topics["targeted_kb_queries"]
        augmentation_source = "web_topics"
        logger.info(f"[Augment] Using {len(targeted_queries)} queries from web search topics")

    # Priority 2: Missing information check (gap-filling queries)
    elif missing_info_check.get("needs_augmentation") and missing_info_check.get("targeted_queries"):
        targeted_queries = missing_info_check["targeted_queries"]
        augmentation_source = "missing_info"
        logger.info(f"[Augment] Using {len(targeted_queries)} queries from missing info check")

    else:
        logger.warning("[Augment] No augmentation queries available, skipping")
        return {
            "context_grounding_results": existing_cg_results,
            "augmentation_source": "none"
        }

    # Execute targeted searches
    new_results = []
    search_limit = config.KNOWLEDGE_BASE_SEARCH_LIMIT

    logger.info(f"[Augment] Executing {len(targeted_queries)} targeted searches:")

    for i, query in enumerate(targeted_queries, 1):
        logger.info(f"[Augment] Query {i}/{len(targeted_queries)}: {query}")

        try:
            # Search Context Grounding
            results = search_knowledge_base(
                query=query,
                number_of_results=search_limit
            )

            logger.info(f"[Augment]   → Found {len(results)} results")

            # Filter by quality (only add results with score > 0.5)
            # Note: results are Pydantic ContextGroundingResult models, not dicts
            quality_threshold = 0.5
            quality_results = [
                r for r in results
                if getattr(r, 'score', 0.0) > quality_threshold
            ]

            if quality_results:
                logger.info(f"[Augment]   → {len(quality_results)} results above quality threshold ({quality_threshold})")
                new_results.extend(quality_results)

                # Log top result details (Pydantic model - use getattr)
                top_result = quality_results[0]
                top_score = getattr(top_result, 'score', 0.0)
                top_source = getattr(top_result, 'source', 'unknown')
                logger.info(f"[Augment]   → Top result: score={top_score:.2f}, source={top_source}")
            else:
                logger.info(f"[Augment]   → No results above quality threshold ({quality_threshold})")

        except Exception as e:
            logger.error(f"[Augment] Failed to search for '{query}': {str(e)}", exc_info=True)
            continue

    # Append new results to existing context grounding results
    augmented_results = existing_cg_results + new_results

    logger.info(f"[Augment] ✓ Augmentation complete")
    logger.info(f"[Augment] Added {len(new_results)} new results")
    logger.info(f"[Augment] Total CG results: {len(existing_cg_results)} → {len(augmented_results)}")
    logger.info(f"[Augment] Source: {augmentation_source}")

    # Deduplicate results by source (keep highest score)
    if augmented_results:
        logger.info("[Augment] Deduplicating results by source...")
        seen_sources = {}

        for result in augmented_results:
            # Handle both Pydantic models and dicts
            if hasattr(result, 'source'):
                source = getattr(result, 'source', 'unknown')
                score = getattr(result, 'score', 0.0)
            else:
                source = result.get("source", "unknown")
                score = result.get("score", 0.0)

            # Keep result with highest score for each source
            if source not in seen_sources:
                seen_sources[source] = result
            else:
                # Compare scores - handle both Pydantic models and dicts
                existing_result = seen_sources[source]
                if hasattr(existing_result, 'score'):
                    # Pydantic model
                    existing_score = existing_result.score if existing_result.score is not None else 0.0
                else:
                    # Dictionary
                    existing_score = existing_result.get("score", 0.0)

                if score > existing_score:
                    seen_sources[source] = result

        deduplicated_results = list(seen_sources.values())
        logger.info(f"[Augment] Deduplicated: {len(augmented_results)} → {len(deduplicated_results)} unique sources")

        augmented_results = deduplicated_results

    return {
        "context_grounding_results": augmented_results,
        "augmentation_source": augmentation_source
    }
