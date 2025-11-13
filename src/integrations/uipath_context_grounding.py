"""
UiPath Context Grounding Integration Module
Handles querying the Context Grounding knowledge base for IT support articles
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from uipath import UiPath
from src.config import config

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class ContextGroundingConfig(BaseModel):
    """Configuration for Context Grounding"""
    index_name: str = "IT"
    folder_path: str = "IT"
    default_number_of_results: int = config.KNOWLEDGE_BASE_SEARCH_LIMIT


class ContextGroundingResult(BaseModel):
    """Model for a single context grounding search result"""
    content: str
    source: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Any] = None  # Can be dict or UiPath metadata object


class ContextGroundingIntegration:
    """Handles UiPath Context Grounding operations for IT knowledge base"""

    def __init__(self, config: Optional[ContextGroundingConfig] = None):
        """
        Initialize Context Grounding integration

        Args:
            config: Optional context grounding configuration, uses defaults if not provided
        """
        self.config = config or ContextGroundingConfig()
        self.sdk = None
        self._initialize_sdk()

    def _initialize_sdk(self):
        """Initialize UiPath SDK with credentials from environment"""
        try:
            self.sdk = UiPath()
            logger.info("UiPath SDK initialized successfully for context grounding")
        except Exception as e:
            logger.error(f"Failed to initialize UiPath SDK: {e}")
            raise

    def search(
        self,
        query: str,
        number_of_results: Optional[int] = None
    ) -> List[ContextGroundingResult]:
        """
        Search the Context Grounding knowledge base

        Args:
            query: Search query text
            number_of_results: Number of results to return (uses config default if not provided)

        Returns:
            List of ContextGroundingResult objects

        Raises:
            Exception: If search fails
        """
        num_results = number_of_results or self.config.default_number_of_results

        logger.info(
            f"Searching context grounding index '{self.config.index_name}' "
            f"with query: {query[:100]}..."
        )

        try:
            # Search the context grounding index
            results = self.sdk.context_grounding.search(
                name=self.config.index_name,
                query=query,
                number_of_results=num_results,
                folder_path=self.config.folder_path
            )

            logger.info(f"Found {len(results)} results from context grounding")

            # Convert to our result model
            formatted_results = []
            for result in results:
                formatted_result = ContextGroundingResult(
                    content=result.content,
                    source=getattr(result, 'source', None),
                    score=getattr(result, 'score', None),
                    metadata=getattr(result, 'metadata', None)
                )
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search context grounding: {e}")
            raise

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the Context Grounding index

        Returns:
            Dict containing index metadata

        Raises:
            Exception: If retrieval fails
        """
        logger.info(f"Retrieving info for index: {self.config.index_name}")

        try:
            index_info = self.sdk.context_grounding.retrieve(
                name=self.config.index_name,
                folder_path=self.config.folder_path
            )

            logger.info(f"Successfully retrieved index info for: {self.config.index_name}")

            # Convert to dict for easier access
            return {
                "name": getattr(index_info, 'name', None),
                "id": getattr(index_info, 'id', None),
                "description": getattr(index_info, 'description', None),
                "created_at": getattr(index_info, 'created_at', None),
                "updated_at": getattr(index_info, 'updated_at', None),
                "document_count": getattr(index_info, 'document_count', None),
            }

        except Exception as e:
            logger.error(f"Failed to retrieve index info: {e}")
            raise


# Convenience functions for LangGraph nodes
def search_knowledge_base(query: str, number_of_results: int = 5) -> List[ContextGroundingResult]:
    """
    Convenience function to search the IT knowledge base

    Args:
        query: Search query text
        number_of_results: Number of results to return

    Returns:
        List of ContextGroundingResult objects
    """
    integration = ContextGroundingIntegration()
    return integration.search(query, number_of_results)


