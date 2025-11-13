"""
UiPath Article Fetching Integration Module
Handles invocation of UiPath processes to fetch published articles from FreshDesk
"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from uipath import UiPath
from src.config import config

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class ArticleInfo(BaseModel):
    """Model for a single article returned from FreshDesk"""
    id: str = Field(description="Article ID")
    title: str = Field(description="Article title")
    description: str = Field(description="Article description/content")
    created_at: str = Field(description="Article creation timestamp")
    updated_at: str = Field(description="Article last update timestamp")


class UiPathArticleJobConfig(BaseModel):
    """Configuration for UiPath article fetching job"""
    process_name: str = Field(
        default=config.UIPATH_ARTICLE_PROCESS_NAME,
        description="Name of the UiPath article fetching process in Orchestrator"
    )
    folder_path: str = Field(
        default=os.getenv("UIPATH_FOLDER_PATH", ""),
        description="UiPath folder path"
    )
    timeout_seconds: int = Field(
        default=config.UIPATH_JOB_TIMEOUT,
        description="Job execution timeout"
    )
    input_keywords_arg: str = Field(
        default=config.UIPATH_INPUT_KEYWORDS,
        description="Input argument name for keywords"
    )
    input_status_arg: str = Field(
        default=config.UIPATH_INPUT_STATUS,
        description="Input argument name for article status"
    )
    default_status: str = Field(
        default=config.UIPATH_DEFAULT_STATUS,
        description="Default status value (e.g., 'published')"
    )
    output_articles_arg: str = Field(
        default=config.UIPATH_OUTPUT_ARTICLES,
        description="Output argument name for articles list"
    )


class UiPathArticleIntegration:
    """Handles UiPath SDK operations for article fetching"""

    def __init__(self, config: Optional[UiPathArticleJobConfig] = None):
        """
        Initialize UiPath article integration

        Args:
            config: Optional job configuration, uses defaults if not provided
        """
        self.config = config or UiPathArticleJobConfig()
        self.sdk = None
        self._initialize_sdk()

    def _initialize_sdk(self):
        """Initialize UiPath SDK with credentials from environment"""
        try:
            self.sdk = UiPath()
            logger.info("UiPath SDK initialized successfully for article fetching")
        except Exception as e:
            logger.error(f"Failed to initialize UiPath SDK: {e}")
            raise

    async def fetch_articles(
        self,
        keywords: str,
        status: Optional[str] = None
    ) -> List[ArticleInfo]:
        """
        Invoke UiPath job to fetch articles from FreshDesk

        Args:
            keywords: Search keywords for articles
            status: Article status filter (default: 'published')

        Returns:
            List of ArticleInfo objects

        Raises:
            Exception: If job invocation fails
        """
        status = status or self.config.default_status
        logger.info(f"Invoking UiPath job to fetch articles with keywords: '{keywords}', status: '{status}'")

        try:
            # Prepare input arguments
            input_args = {
                self.config.input_keywords_arg: keywords,
                self.config.input_status_arg: status
            }

            logger.debug(f"Input arguments: {input_args}")

            # Invoke the UiPath process
            job = self.sdk.processes.invoke(
                name=self.config.process_name,
                input_arguments=input_args,
                folder_path=self.config.folder_path if self.config.folder_path else None
            )

            logger.info(f"UiPath article fetching job started with key: {job.key}")

            # Wait for job completion
            job_result = self._wait_for_job_completion(job.key)

            # Parse articles from job output
            articles = self._parse_job_output(job_result)

            logger.info(f"Successfully retrieved {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch articles with keywords '{keywords}': {e}")
            raise

    def _wait_for_job_completion(self, job_key: str) -> Dict[str, Any]:
        """
        Wait for UiPath job to complete

        Args:
            job_key: The UiPath job key

        Returns:
            Job result with output arguments
        """
        max_attempts = self.config.timeout_seconds
        attempt = 0

        while attempt < max_attempts:
            try:
                job_status = self.sdk.jobs.retrieve(
                    job_key,
                    folder_path=self.config.folder_path if self.config.folder_path else None
                )

                if job_status.state == "Successful":
                    logger.info(f"Job {job_key} completed successfully")
                    return job_status.output_arguments or {}

                elif job_status.state in ["Faulted", "Stopped"]:
                    error_msg = f"Job {job_key} failed with state: {job_status.state}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                logger.debug(f"Job {job_key} state: {job_status.state}, waiting...")
                time.sleep(1)
                attempt += 1

            except Exception as e:
                logger.error(f"Error checking job status: {e}")
                raise

        raise TimeoutError(f"Job {job_key} did not complete within {self.config.timeout_seconds} seconds")

    def _parse_job_output(self, output_args: Any) -> List[ArticleInfo]:
        """
        Parse UiPath job output into list of ArticleInfo objects

        Expected format: List<string> where each string is:
        "id: 123 --- title: Title --- description: Desc --- created_at: Date --- updated_at: Date"

        Args:
            output_args: Output arguments from UiPath job

        Returns:
            List of ArticleInfo objects
        """
        try:
            # Parse output_args if it's a JSON string
            if isinstance(output_args, str):
                logger.debug("Output arguments received as string, parsing JSON...")
                output_args = json.loads(output_args)

            # Get the articles list from output
            articles_raw = output_args.get(self.config.output_articles_arg)

            if not articles_raw:
                logger.warning(f"No articles found in output argument '{self.config.output_articles_arg}'")
                return []

            # Parse each article string
            articles = []
            for article_str in articles_raw:
                try:
                    article_info = self._parse_article_string(article_str)
                    articles.append(article_info)
                except Exception as e:
                    logger.error(f"Failed to parse article string: {e}")
                    logger.error(f"Article string: {article_str[:200]}...")  # Log first 200 chars
                    continue

            logger.info(f"Successfully parsed {len(articles)} articles")
            return articles

        except Exception as e:
            logger.error(f"Failed to parse job output: {e}")
            logger.error(f"Output arguments received: {output_args}")
            raise

    def _parse_article_string(self, article_str: str) -> ArticleInfo:
        """
        Parse a single article string into ArticleInfo object

        Format: "id: 123 --- title: Title --- description: Desc --- created_at: Date --- updated_at: Date"

        Args:
            article_str: Article string from UiPath

        Returns:
            ArticleInfo object
        """
        # Split by " --- " delimiter
        parts = article_str.split(" --- ")

        # Initialize article data
        article_data = {
            "id": "",
            "title": "",
            "description": "",
            "created_at": "",
            "updated_at": ""
        }

        # Parse each part
        for part in parts:
            # Split on first ": " to separate key from value
            if ": " in part:
                key, value = part.split(": ", 1)
                key = key.strip().lower().replace(" ", "_")

                if key in article_data:
                    article_data[key] = value.strip()

        # Validate required fields
        if not article_data["id"]:
            raise ValueError("Article ID is missing")

        return ArticleInfo(**article_data)


# Convenience function for LangGraph node
async def fetch_articles_from_uipath(
    keywords: str,
    status: Optional[str] = None
) -> List[ArticleInfo]:
    """
    Convenience function to fetch articles from UiPath

    Args:
        keywords: Search keywords
        status: Article status filter (default: 'published')

    Returns:
        List of ArticleInfo objects
    """
    integration = UiPathArticleIntegration()
    return await integration.fetch_articles(keywords, status)