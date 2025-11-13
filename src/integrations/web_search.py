"""
Web Search Integration Module
Secure DuckDuckGo search with domain whitelisting and content sanitization
"""

import logging
import re
import time
from typing import List, Optional
from urllib.parse import urlparse
import httpx
from pydantic import BaseModel, Field

from src.config import config

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Model for a single search result"""
    title: str = Field(description="Page title")
    url: str = Field(description="URL of the page")
    content: str = Field(description="Snippet/description of the page")
    source_domain: str = Field(description="Domain name of the source")


class WebSearchIntegration:
    """Handles secure web searches using DuckDuckGo with whitelist filtering"""

    def __init__(
        self,
        trusted_domains: Optional[List[str]] = None,
        max_results: int = None,
        timeout: int = None
    ):
        """
        Initialize web search integration

        Args:
            trusted_domains: List of trusted domains (uses config if None)
            max_results: Maximum results to return (uses config if None)
            timeout: Timeout in seconds (uses config if None)
        """
        self.trusted_domains = trusted_domains or config.WEB_SEARCH_TRUSTED_DOMAINS
        self.max_results = max_results or config.WEB_SEARCH_MAX_RESULTS
        self.timeout = timeout or config.WEB_SEARCH_TIMEOUT_SECONDS
        self.rate_limit_delay = config.WEB_SEARCH_RATE_LIMIT_DELAY
        self.max_content_length = config.WEB_SEARCH_MAX_CONTENT_LENGTH

        logger.info(f"WebSearch initialized with {len(self.trusted_domains)} trusted domains")

    async def search(self, query: str, max_results: int = None) -> List[SearchResult]:
        """
        Perform secure web search with domain whitelisting

        Args:
            query: Search query string
            max_results: Override max results (optional)

        Returns:
            List of filtered SearchResult objects

        Raises:
            Exception: If search fails
        """
        max_results = max_results or self.max_results

        logger.info(f"Performing web search: '{query}' (max: {max_results})")

        try:
            # Add rate limiting
            time.sleep(self.rate_limit_delay)

            # Perform DuckDuckGo search
            raw_results = await self._duckduckgo_search(query, max_results)

            logger.debug(f"Got {len(raw_results)} raw results from DuckDuckGo")

            # Filter by trusted domains
            filtered_results = self._filter_by_trusted_domains(raw_results)

            logger.info(f"Filtered to {len(filtered_results)} results from trusted domains")

            # Sanitize content
            sanitized_results = [self._sanitize_result(r) for r in filtered_results]

            # Limit to max_results
            final_results = sanitized_results[:max_results]

            logger.info(f"Returning {len(final_results)} sanitized results")

            return final_results

        except Exception as e:
            logger.error(f"Web search failed: {e}", exc_info=True)
            raise

    async def _duckduckgo_search(self, query: str, max_results: int) -> List[dict]:
        """
        Perform DuckDuckGo search using their instant answer API

        Args:
            query: Search query
            max_results: Maximum results to fetch

        Returns:
            List of raw result dictionaries
        """
        try:
            # DuckDuckGo HTML search endpoint (no API key required)
            url = "https://html.duckduckgo.com/html/"

            # Prepare request
            params = {"q": query}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, data=params, headers=headers)
                response.raise_for_status()

                # Parse HTML response
                results = self._parse_duckduckgo_html(response.text, max_results)

                return results

        except httpx.TimeoutException:
            logger.error(f"DuckDuckGo search timed out after {self.timeout}s")
            raise Exception(f"Search timeout after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            raise Exception(f"Search failed: {str(e)}")

    def _parse_duckduckgo_html(self, html: str, max_results: int) -> List[dict]:
        """
        Parse DuckDuckGo HTML response to extract search results

        Args:
            html: HTML response from DuckDuckGo
            max_results: Maximum results to extract

        Returns:
            List of result dictionaries
        """
        results = []

        try:
            # Simple regex-based parsing (basic but works)
            # Match result blocks
            result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'

            # Find all result links
            link_matches = re.findall(result_pattern, html)

            for idx, (url, title) in enumerate(link_matches[:max_results * 2]):  # Get extra for filtering
                # Decode URL
                url = url.replace("&amp;", "&")

                # Extract snippet (if available)
                # Look for snippet near this result
                snippet_start = html.find(title)
                if snippet_start != -1:
                    snippet_section = html[snippet_start:snippet_start + 500]
                    snippet_match = re.search(snippet_pattern, snippet_section)
                    content = snippet_match.group(1) if snippet_match else title
                else:
                    content = title

                results.append({
                    "title": self._clean_html(title),
                    "url": url,
                    "content": self._clean_html(content)
                })

                if len(results) >= max_results * 2:  # Extra for filtering
                    break

            logger.debug(f"Parsed {len(results)} results from HTML")
            return results

        except Exception as e:
            logger.error(f"Failed to parse DuckDuckGo HTML: {e}")
            return []

    def _filter_by_trusted_domains(self, results: List[dict]) -> List[SearchResult]:
        """
        Filter results to only include trusted domains

        Args:
            results: List of raw result dictionaries

        Returns:
            List of SearchResult objects from trusted domains
        """
        filtered = []

        for result in results:
            try:
                url = result.get("url", "")
                domain = self._extract_domain(url)

                # Check if domain is trusted
                if self._is_trusted_domain(domain):
                    filtered.append(SearchResult(
                        title=result.get("title", ""),
                        url=url,
                        content=result.get("content", ""),
                        source_domain=domain
                    ))
                else:
                    logger.debug(f"Filtered out untrusted domain: {domain}")

            except Exception as e:
                logger.warning(f"Error filtering result: {e}")
                continue

        return filtered

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL

        Args:
            url: Full URL

        Returns:
            Domain name (e.g., "microsoft.com")
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            return domain
        except Exception:
            return ""

    def _is_trusted_domain(self, domain: str) -> bool:
        """
        Check if domain is in trusted list

        Args:
            domain: Domain to check

        Returns:
            True if trusted
        """
        # Check exact match
        if domain in self.trusted_domains:
            return True

        # Check if it's a subdomain of a trusted domain
        for trusted in self.trusted_domains:
            if domain.endswith(f".{trusted}") or domain == trusted:
                return True

        return False

    def _sanitize_result(self, result: SearchResult) -> SearchResult:
        """
        Sanitize search result content

        Args:
            result: SearchResult to sanitize

        Returns:
            Sanitized SearchResult
        """
        # Sanitize title
        result.title = self._sanitize_text(result.title)

        # Sanitize content and limit length
        result.content = self._sanitize_text(result.content)
        if len(result.content) > self.max_content_length:
            result.content = result.content[:self.max_content_length] + "..."

        return result

    def _sanitize_text(self, text: str) -> str:
        """
        Remove potentially dangerous content from text

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        # Remove HTML tags
        text = self._clean_html(text)

        # Remove any remaining suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                 # JavaScript URLs
            r'on\w+\s*=',                  # Event handlers
        ]

        for pattern in suspicious_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _clean_html(self, text: str) -> str:
        """
        Remove HTML tags from text

        Args:
            text: Text with HTML

        Returns:
            Clean text
        """
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)

        # Decode HTML entities
        clean = clean.replace('&amp;', '&')
        clean = clean.replace('&lt;', '<')
        clean = clean.replace('&gt;', '>')
        clean = clean.replace('&quot;', '"')
        clean = clean.replace('&#39;', "'")
        clean = clean.replace('&nbsp;', ' ')

        return clean


# Convenience function for quick searches
async def search_web(query: str, max_results: int = None) -> List[SearchResult]:
    """
    Convenience function to perform web search

    Args:
        query: Search query
        max_results: Maximum results (uses config if None)

    Returns:
        List of SearchResult objects
    """
    integration = WebSearchIntegration()
    return await integration.search(query, max_results)