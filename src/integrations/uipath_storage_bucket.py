"""
UiPath Storage Bucket Integration Module
Handles downloading and parsing IT actions from UiPath storage buckets
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from uipath import UiPath
from src.config import config

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class ITActionsConfig(BaseModel):
    """Configuration for IT Actions storage"""
    bucket_name: str = config.UIPATH_BUCKET_NAME
    file_path: str = config.IT_ACTIONS_JSON_PATH
    folder_path: str = os.getenv("UIPATH_FOLDER_PATH", "")


class StorageBucketIntegration:
    """Handles UiPath Storage Bucket operations for IT Actions"""

    def __init__(self, config: Optional[ITActionsConfig] = None):
        """
        Initialize Storage Bucket integration

        Args:
            config: Optional storage configuration, uses defaults if not provided
        """
        self.config = config or ITActionsConfig()
        self.sdk = None
        self._initialize_sdk()

    def _initialize_sdk(self):
        """Initialize UiPath SDK with credentials from environment"""
        try:
            self.sdk = UiPath()
            logger.info("UiPath SDK initialized successfully for storage bucket")
        except Exception as e:
            logger.error(f"Failed to initialize UiPath SDK: {e}")
            raise

    def load_it_actions(self) -> Dict[str, Any]:
        """
        Download and parse IT actions JSON from UiPath storage bucket

        Returns:
            Dict containing IT actions configuration

        Raises:
            Exception: If download or parsing fails
        """
        logger.info(f"Loading IT actions from bucket: {self.config.bucket_name}/{self.config.file_path}")

        try:
            # Create temporary file path for download
            temp_file = "temp_it_actions.json"

            # Download from bucket
            self.sdk.buckets.download(
                name=self.config.bucket_name,
                blob_file_path=self.config.file_path,
                destination_path=temp_file,
                folder_path=self.config.folder_path if self.config.folder_path else None
            )

            logger.info(f"Downloaded IT actions file to: {temp_file}")

            # Read and parse JSON
            with open(temp_file, 'r', encoding='utf-8') as f:
                it_actions_data = json.load(f)

            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug("Cleaned up temporary file")

            logger.info(f"Successfully loaded IT actions with {len(it_actions_data.get('it_human_actions', {}))} categories")
            return it_actions_data

        except Exception as e:
            logger.error(f"Failed to load IT actions: {e}")
            raise

    def get_it_action_categories(self) -> Dict[str, Any]:
        """
        Get all IT action categories from storage

        Returns:
            Dict of IT action categories
        """
        it_actions_data = self.load_it_actions()
        return it_actions_data.get('it_human_actions', {})

    def get_classification_config(self) -> Dict[str, Any]:
        """
        Get classification configuration from storage

        Returns:
            Dict containing classification thresholds and settings
        """
        it_actions_data = self.load_it_actions()
        return it_actions_data.get('classification_config', {})


# Convenience function for LangGraph nodes
def fetch_it_actions_from_bucket() -> Dict[str, Any]:
    """
    Convenience function to fetch IT actions from UiPath storage bucket

    Returns:
        Dict containing complete IT actions configuration
    """
    integration = StorageBucketIntegration()
    return integration.load_it_actions()
