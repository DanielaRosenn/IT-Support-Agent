"""
Unified Article Search Interface

Provides a single interface that can use either:
1. Local cache (fast, free) - DEFAULT
2. UiPath/FreshService API (fallback)

Usage:
    from src.utils.article_search import search_articles_unified

    # Uses local cache by default
    articles = await search_articles_unified("password reset", limit=5)

    # Force UiPath/FreshService search
    articles = await search_articles_unified("password reset", use_cache=False)
"""

import logging
from typing import List, Optional
from src.utils.article_cache import get_article_cache, ArticleSearchResult
from src.integrations.uipath_fetch_articles import fetch_articles_from_uipath, ArticleInfo

logger = logging.getLogger(__name__)


async def search_articles_unified(
    keywords: str,
    status: str = "published",
    limit: int = 10,
    use_cache: bool = True,
    cache_fallback_to_uipath: bool = True
) -> List[ArticleInfo]:
    """
    Unified article search - uses local cache by default, with UiPath fallback

    Args:
        keywords: Search keywords
        status: Article status filter (default: published)
        limit: Maximum number of results
        use_cache: Whether to use local cache (default: True)
        cache_fallback_to_uipath: If cache fails, fallback to UiPath (default: True)

    Returns:
        List of ArticleInfo objects

    Example:
        # Fast local search (recommended)
        articles = await search_articles_unified("password reset sspr", limit=5)

        # Force UiPath/FreshService search
        articles = await search_articles_unified("password reset", use_cache=False)
    """
    if use_cache:
        try:
            # Try local cache first
            logger.info(f"Searching local cache for: {keywords}")
            cache = get_article_cache()

            # Check if cache is available
            stats = cache.get_stats()
            if not stats['cache_exists'] or stats['total_articles'] == 0:
                logger.warning("Local cache not available or empty")
                if cache_fallback_to_uipath:
                    logger.info("Falling back to UiPath/FreshService search")
                    return await fetch_articles_from_uipath(keywords, status)
                else:
                    logger.warning("Cache fallback disabled, returning empty results")
                    return []

            # Search local cache
            results = cache.search(keywords, status=status, limit=limit)

            logger.info(f"Local cache returned {len(results)} results")

            # Convert to ArticleInfo format for compatibility
            articles = []
            for result in results:
                articles.append(ArticleInfo(
                    id=result.article.id,
                    title=result.article.title,
                    description=result.article.description,
                    created_at=result.article.created_at,
                    updated_at=result.article.updated_at
                ))

            # Log search quality
            if results:
                avg_score = sum(r.score for r in results) / len(results)
                logger.info(f"Average relevance score: {avg_score:.2f}")

                # If average score is very low, warn about cache quality
                if avg_score < 3.0 and cache_fallback_to_uipath:
                    logger.warning(f"Low relevance scores (avg: {avg_score:.2f})")
                    logger.warning("Consider re-syncing cache or using UiPath search")

            return articles

        except Exception as e:
            logger.error(f"Local cache search failed: {e}", exc_info=True)

            if cache_fallback_to_uipath:
                logger.info("Falling back to UiPath/FreshService search due to cache error")
                return await fetch_articles_from_uipath(keywords, status)
            else:
                raise

    else:
        # Use UiPath/FreshService directly
        logger.info(f"Using UiPath/FreshService search for: {keywords}")
        return await fetch_articles_from_uipath(keywords, status)


def get_cache_stats() -> dict:
    """
    Get statistics about the local article cache

    Returns:
        Dictionary with cache statistics

    Example:
        stats = get_cache_stats()
        print(f"Cache has {stats['total_articles']} articles")
        print(f"Last synced: {stats['last_synced']}")
    """
    cache = get_article_cache()
    return cache.get_stats()


def is_cache_available() -> bool:
    """
    Check if local cache is available and populated

    Returns:
        True if cache exists and has articles, False otherwise

    Example:
        if is_cache_available():
            print("Using fast local search")
        else:
            print("Cache not available, run: python scripts/sync_articles.py")
    """
    stats = get_cache_stats()
    return stats['cache_exists'] and stats['total_articles'] > 0