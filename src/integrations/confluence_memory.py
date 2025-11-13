"""
Confluence Memory Integration Module
Handles storing successful ticket resolutions to Confluence and searching them via Context Grounding
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from uipath import UiPath
from src.config import config

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class ConfluenceMemoryConfig(BaseModel):
    """Configuration for Confluence Memory storage and retrieval"""
    index_name: str = "IT Approved Tickets"
    folder_path: str = "IT/IT01_support_agent_fresh"
    default_number_of_results: int = 3


class TicketResolution(BaseModel):
    """Model for a ticket resolution to be stored in Confluence"""
    ticket_id: str
    category: str
    subject: str
    requester: str
    date: str
    priority: str
    description: str
    response: str
    related_articles: List[Dict[str, str]]  # List of {id: str, title: str}
    status: str = "successful"


class ConfluenceMemoryResult(BaseModel):
    """Model for a single memory search result from Context Grounding"""
    content: str
    source: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Any] = None


class ConfluenceMemoryIntegration:
    """Handles storing and retrieving ticket resolutions from Confluence"""

    def __init__(self, config: Optional[ConfluenceMemoryConfig] = None):
        """
        Initialize Confluence Memory integration

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or ConfluenceMemoryConfig()
        self.sdk = None
        self._initialize_sdk()

    def _initialize_sdk(self):
        """Initialize UiPath SDK with credentials from environment"""
        try:
            self.sdk = UiPath()
            logger.info("UiPath SDK initialized successfully for Confluence memory")
        except Exception as e:
            logger.error(f"Failed to initialize UiPath SDK: {e}")
            raise

    def create_confluence_page(
        self,
        resolution: TicketResolution
    ) -> Dict[str, Any]:
        """
        Create a Confluence page for a successful ticket resolution

        Args:
            resolution: TicketResolution object containing all ticket details

        Returns:
            Dict with page creation result including page_id and url

        Raises:
            Exception: If page creation fails
        """
        logger.info(f"Creating Confluence page for ticket #{resolution.ticket_id}")

        try:
            # Generate page title
            page_title = f"Ticket #{resolution.ticket_id} - {resolution.subject}"

            # Generate page content in markdown format
            page_content = self._generate_page_content(resolution)

            # Create metadata for Context Grounding
            metadata = {
                "ticket_id": resolution.ticket_id,
                "category": resolution.category,
                "status": resolution.status
            }

            # Create the Confluence page using UiPath SDK
            # Note: This assumes UiPath SDK has a confluence integration
            # You may need to adjust based on actual SDK capabilities
            result = self.sdk.confluence.create_page(
                space_key=self.config.folder_path.split("/")[0],  # "IT"
                title=page_title,
                content=page_content,
                parent_id=None,  # Optional: set parent page ID if needed
                metadata=metadata
            )

            logger.info(f"Successfully created Confluence page for ticket #{resolution.ticket_id}")

            return {
                "page_id": result.get("id"),
                "url": result.get("url"),
                "title": page_title,
                "ticket_id": resolution.ticket_id
            }

        except Exception as e:
            logger.error(f"Failed to create Confluence page for ticket #{resolution.ticket_id}: {e}")
            raise

    def _generate_page_content(self, resolution: TicketResolution) -> str:
        """
        Generate markdown content for Confluence page

        Args:
            resolution: TicketResolution object

        Returns:
            Formatted markdown string
        """
        # Format related articles
        articles_section = ""
        if resolution.related_articles:
            articles_section = "## Related Articles\n\nFrom FreshService:\n"
            for article in resolution.related_articles:
                articles_section += f"- Article #{article['id']}: {article['title']}\n"
            articles_section += "\n---\n\n"

        # Format article IDs for metadata
        article_ids = [article['id'] for article in resolution.related_articles] if resolution.related_articles else []

        content = f"""# Ticket #{resolution.ticket_id} - {resolution.subject}

## Ticket Information

**Category**: {resolution.category}
**Subject**: {resolution.subject}
**Requester**: {resolution.requester}
**Date**: {resolution.date}
**Priority**: {resolution.priority}

**Description**:
{resolution.description}

---

## Response

{resolution.response}

---

{articles_section}## Metadata
```json
{{
  "ticket_id": "{resolution.ticket_id}",
  "category": "{resolution.category}",
  "status": "{resolution.status}"{', "article_ids": ' + str(article_ids).replace("'", '"') if article_ids else ''}
}}
```
"""
        return content

    def search_similar_resolutions(
        self,
        query: str,
        number_of_results: Optional[int] = None
    ) -> List[ConfluenceMemoryResult]:
        """
        Search for similar ticket resolutions using Context Grounding

        Args:
            query: Search query (typically the ticket description or category)
            number_of_results: Number of results to return (uses config default if not provided)

        Returns:
            List of ConfluenceMemoryResult objects containing similar past resolutions

        Raises:
            Exception: If search fails
        """
        num_results = number_of_results or self.config.default_number_of_results

        logger.info(
            f"Searching memory index '{self.config.index_name}' "
            f"with query: {query[:100]}..."
        )

        try:
            # Search the Context Grounding index for past resolutions
            results = self.sdk.context_grounding.search(
                name=self.config.index_name,
                query=query,
                number_of_results=num_results,
                folder_path=self.config.folder_path
            )

            logger.info(f"Found {len(results)} similar resolutions from memory")

            # Convert to our result model
            formatted_results = []
            for result in results:
                formatted_result = ConfluenceMemoryResult(
                    content=result.content,
                    source=getattr(result, 'source', None),
                    score=getattr(result, 'score', None),
                    metadata=getattr(result, 'metadata', None)
                )
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search memory index: {e}")
            raise

    def get_memory_index_info(self) -> Dict[str, Any]:
        """
        Get information about the Confluence memory index

        Returns:
            Dict containing index metadata

        Raises:
            Exception: If retrieval fails
        """
        logger.info(f"Retrieving info for memory index: {self.config.index_name}")

        try:
            index_info = self.sdk.context_grounding.retrieve(
                name=self.config.index_name,
                folder_path=self.config.folder_path
            )

            logger.info(f"Successfully retrieved memory index info")

            return {
                "name": getattr(index_info, 'name', None),
                "id": getattr(index_info, 'id', None),
                "description": getattr(index_info, 'description', None),
                "created_at": getattr(index_info, 'created_at', None),
                "updated_at": getattr(index_info, 'updated_at', None),
                "document_count": getattr(index_info, 'document_count', None),
            }

        except Exception as e:
            logger.error(f"Failed to retrieve memory index info: {e}")
            raise


# Convenience functions for LangGraph nodes
def store_successful_resolution(resolution: TicketResolution) -> Dict[str, Any]:
    """
    Convenience function to store a successful ticket resolution to Confluence

    Args:
        resolution: TicketResolution object

    Returns:
        Dict with page creation result
    """
    integration = ConfluenceMemoryIntegration()
    return integration.create_confluence_page(resolution)


def search_past_resolutions(query: str, number_of_results: int = 3) -> List[ConfluenceMemoryResult]:
    """
    Convenience function to search for similar past resolutions

    Args:
        query: Search query text
        number_of_results: Number of results to return

    Returns:
        List of ConfluenceMemoryResult objects
    """
    integration = ConfluenceMemoryIntegration()
    return integration.search_similar_resolutions(query, number_of_results)


def get_memory_stats() -> Dict[str, Any]:
    """
    Convenience function to get memory index statistics

    Returns:
        Dict containing index metadata
    """
    integration = ConfluenceMemoryIntegration()
    return integration.get_memory_index_info()