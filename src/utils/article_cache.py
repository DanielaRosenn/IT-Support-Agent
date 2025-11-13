"""
Local Article Cache for Fast Search
Provides in-memory search over cached FreshService articles
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CachedArticle(BaseModel):
    """Model for a cached article"""
    id: str = Field(description="Article ID")
    title: str = Field(description="Article title")
    description: str = Field(description="Article description/content")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Last update timestamp")
    status: str = Field(default="published", description="Article status")


class ArticleSearchResult(BaseModel):
    """Model for search result with score"""
    article: CachedArticle
    score: float = Field(description="Relevance score")
    match_details: Dict[str, int] = Field(
        default_factory=dict,
        description="Details about what matched (title_matches, description_matches, etc.)"
    )


class LocalArticleCache:
    """
    Local article cache with simple keyword-based search

    Fast, free, and controllable alternative to FreshService API search
    """

    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize article cache

        Args:
            cache_file: Path to cache JSON file (default: docs/article_cache.json)
        """
        if cache_file is None:
            # Default to docs/article_cache.json relative to project root
            project_root = Path(__file__).parent.parent.parent
            cache_file = project_root / "docs" / "article_cache.json"

        self.cache_file = cache_file
        self.articles: List[CachedArticle] = []
        self.last_synced: Optional[str] = None
        self._load_cache()

    def _load_cache(self):
        """Load articles from cache file"""
        if not self.cache_file.exists():
            logger.warning(f"Cache file not found: {self.cache_file}")
            logger.warning("Run 'python scripts/sync_articles.py' to create cache")
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.last_synced = data.get('last_synced')
            self.articles = [CachedArticle(**article) for article in data.get('articles', [])]

            logger.info(f"Loaded {len(self.articles)} articles from cache")
            logger.info(f"Cache last synced: {self.last_synced}")

        except Exception as e:
            logger.error(f"Error loading cache: {e}", exc_info=True)
            self.articles = []

    def search(
        self,
        keywords: str,
        status: str = "published",
        limit: int = 10
    ) -> List[ArticleSearchResult]:
        """
        Search articles by keywords with scoring

        Scoring algorithm:
        - Title exact phrase match: +10 points
        - Title keyword match: +3 points per keyword
        - Description keyword match: +1 point per keyword
        - Keyword density bonus: +5 if >50% keywords match

        Args:
            keywords: Space-separated search keywords
            status: Filter by article status (default: published)
            limit: Maximum number of results to return

        Returns:
            List of ArticleSearchResult, sorted by score (highest first)

        Example:
            cache = LocalArticleCache()
            results = cache.search("password reset jump host", limit=5)
            for result in results:
                print(f"{result.article.title} (score: {result.score})")
        """
        if not self.articles:
            logger.warning("No articles in cache - returning empty results")
            return []

        # Filter by status
        filtered_articles = [a for a in self.articles if a.status == status]

        if not filtered_articles:
            logger.warning(f"No articles with status '{status}'")
            return []

        # Normalize keywords
        keywords_lower = keywords.lower().strip()
        keyword_list = keywords_lower.split()

        if not keyword_list:
            logger.warning("No keywords provided")
            return []

        logger.debug(f"Searching {len(filtered_articles)} articles for: {keywords_lower}")

        # Score each article
        results = []
        for article in filtered_articles:
            score, match_details = self._score_article(article, keywords_lower, keyword_list)

            if score > 0:
                results.append(ArticleSearchResult(
                    article=article,
                    score=score,
                    match_details=match_details
                ))

        # Sort by score (highest first) and limit
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Found {len(results)} matching articles, returning top {limit}")

        return results[:limit]

    def _score_article(
        self,
        article: CachedArticle,
        keywords_phrase: str,
        keyword_list: List[str]
    ) -> tuple[float, Dict[str, int]]:
        """
        Score an article's relevance to keywords

        Args:
            article: Article to score
            keywords_phrase: Full keyword phrase (lowercase)
            keyword_list: Individual keywords (lowercase)

        Returns:
            Tuple of (score, match_details)
        """
        score = 0.0
        match_details = {
            "title_exact_match": 0,
            "title_keyword_matches": 0,
            "description_keyword_matches": 0,
            "density_bonus": 0,
            "all_keywords_in_title": 0
        }

        title_lower = article.title.lower()
        description_lower = article.description.lower()

        # 1. Exact phrase match in title (highest weight)
        if keywords_phrase in title_lower:
            score += 15.0  # Increased from 10
            match_details["title_exact_match"] = 1

        # 2. Individual keyword matches in title (high weight)
        title_matches = 0
        for keyword in keyword_list:
            if keyword in title_lower:
                score += 5.0  # Increased from 3
                title_matches += 1
        match_details["title_keyword_matches"] = title_matches

        # 3. Individual keyword matches in description (low weight)
        description_matches = 0
        for keyword in keyword_list:
            if keyword in description_lower:
                score += 0.5  # Decreased from 1
                description_matches += 1
        match_details["description_keyword_matches"] = description_matches

        # 4. ALL keywords in title bonus (very important)
        if title_matches == len(keyword_list) and len(keyword_list) > 1:
            score += 10.0  # NEW: Big bonus if title has all keywords
            match_details["all_keywords_in_title"] = 1

        # 5. High keyword density bonus
        total_matches = title_matches + description_matches
        if total_matches >= len(keyword_list) * 0.7:  # Increased threshold from 50% to 70%
            score += 3.0  # Decreased from 5
            match_details["density_bonus"] = 1

        return score, match_details

    def get_article_by_id(self, article_id: str) -> Optional[CachedArticle]:
        """
        Get a specific article by ID

        Args:
            article_id: Article ID to find

        Returns:
            CachedArticle if found, None otherwise
        """
        for article in self.articles:
            if article.id == article_id:
                return article
        return None

    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        return {
            "total_articles": len(self.articles),
            "last_synced": self.last_synced,
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists()
        }


# Global cache instance
_cache = None


def get_article_cache() -> LocalArticleCache:
    """
    Get global article cache instance (singleton pattern)

    Returns:
        LocalArticleCache instance
    """
    global _cache
    if _cache is None:
        _cache = LocalArticleCache()
    return _cache


# Convenience function for direct search
def search_articles(keywords: str, status: str = "published", limit: int = 10) -> List[ArticleSearchResult]:
    """
    Convenience function to search articles

    Args:
        keywords: Search keywords
        status: Article status filter
        limit: Maximum results

    Returns:
        List of ArticleSearchResult

    Example:
        from src.utils.article_cache import search_articles

        results = search_articles("password reset sspr", limit=5)
        for result in results:
            print(f"{result.article.title}: {result.score}")
    """
    cache = get_article_cache()
    return cache.search(keywords, status, limit)