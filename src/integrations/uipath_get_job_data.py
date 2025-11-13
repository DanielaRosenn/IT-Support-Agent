"""
UiPath Job Integration Module
Handles invocation of UiPath processes and job management
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, create_model
from uipath import UiPath
from src.config import config
from src.utils.exceptions import IntegrationError, ConfigurationError, ValidationError, wrap_integration_error

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# Dynamically create TicketInfo model from config
def _create_ticket_info_model():
    """
    Dynamically create TicketInfo Pydantic model based on config.UIPATH_OUTPUT_MAPPING
    This allows fields to be added/removed just by updating the config file

    All fields (except ticket_id) are Optional[str] to gracefully handle None values
    from UiPath. None values will be converted to defaults during parsing.
    """
    from typing import Optional

    # Start with ticket_id which is always required
    field_definitions = {
        'ticket_id': (str, ...),  # Required field
    }

    # Add all fields from output mapping dynamically as Optional[str]
    # This allows None values from UiPath without validation errors
    for field_name in config.UIPATH_OUTPUT_MAPPING.keys():
        default_value = config.UIPATH_OUTPUT_DEFAULTS.get(field_name, None)
        # Use Optional[str] to allow None values, with default fallback
        field_definitions[field_name] = (Optional[str], default_value)

    # Create the model dynamically
    return create_model(
        'TicketInfo',
        __doc__='Model for ticket information returned from UiPath job (dynamically generated from config)',
        **field_definitions
    )


# Create the TicketInfo class from config
TicketInfo = _create_ticket_info_model()


class UiPathJobConfig(BaseModel):
    """Configuration for UiPath job invocation"""
    process_name: str = Field(
        default=config.UIPATH_PROCESS_NAME,
        description="Name of the UiPath process in Orchestrator"
    )
    folder_path: str = Field(
        default=os.getenv("UIPATH_FOLDER_PATH", ""),
        description="UiPath folder path"
    )
    timeout_seconds: int = Field(
        default=config.UIPATH_JOB_TIMEOUT,
        description="Job execution timeout"
    )
    input_ticket_id_arg: str = Field(
        default=config.UIPATH_INPUT_TICKET_ID,
        description="Input argument name for ticket ID"
    )
    output_mapping: Dict[str, str] = Field(
        default=config.UIPATH_OUTPUT_MAPPING,
        description="Mapping of internal field names to UiPath output arguments"
    )
    output_defaults: Dict[str, Any] = Field(
        default=config.UIPATH_OUTPUT_DEFAULTS,
        description="Default values for missing output arguments"
    )


class UiPathIntegration:
    """Handles UiPath SDK operations for ticket processing"""

    def __init__(self, config: Optional[UiPathJobConfig] = None):
        """
        Initialize UiPath integration

        Args:
            config: Optional job configuration, uses defaults if not provided
        """
        self.config = config or UiPathJobConfig()
        self.sdk = None
        self._initialize_sdk()

    def _initialize_sdk(self):
        """Initialize UiPath SDK with credentials from environment"""
        try:
            # UiPath SDK will automatically read from environment variables:
            # UIPATH_BASE_URL, UIPATH_ACCESS_TOKEN, UIPATH_ORGANIZATION_ID, etc.
            self.sdk = UiPath()
            logger.info("UiPath SDK initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize UiPath SDK: {e}")
            raise ConfigurationError(
                "UiPath SDK initialization failed - check credentials",
                details={
                    "error": str(e),
                    "required_env_vars": [
                        "UIPATH_URL",
                        "UIPATH_ACCESS_TOKEN",
                        "UIPATH_ORGANIZATION_ID",
                        "UIPATH_TENANT_ID"
                    ]
                }
            )

    async def get_ticket_info(self, ticket_id: str) -> TicketInfo:
        """
        Invoke UiPath job to retrieve ticket information

        Args:
            ticket_id: The ticket ID to retrieve information for

        Returns:
            TicketInfo object with all ticket details

        Raises:
            Exception: If job invocation fails
        """
        logger.info(f"Invoking UiPath job for ticket: {ticket_id}")

        try:
            # Prepare input arguments for UiPath job using configured input name
            input_args = {
                self.config.input_ticket_id_arg: ticket_id
            }

            logger.debug(f"Input arguments: {input_args}")

            # Invoke the UiPath process
            job = self.sdk.processes.invoke(
                name=self.config.process_name,
                input_arguments=input_args,
                folder_path=self.config.folder_path if self.config.folder_path else None
            )

            logger.info(f"UiPath job started with key: {job.key}")

            # Wait for job completion (with timeout)
            job_result = self._wait_for_job_completion(job.key)

            # Extract outputs from job result
            ticket_info = self._parse_job_output(job_result, ticket_id)

            logger.info(f"Successfully retrieved ticket info for: {ticket_id}")
            return ticket_info

        except TimeoutError as e:
            # Job timeout - specific integration error
            raise wrap_integration_error(
                exception=e,
                service="UiPath Processes",
                operation=f"get_ticket_info({ticket_id})"
            )

        except Exception as e:
            # Generic integration failure
            logger.error(f"Failed to get ticket info for {ticket_id}: {e}", exc_info=True)
            raise wrap_integration_error(
                exception=e,
                service="UiPath Processes",
                operation=f"get_ticket_info({ticket_id})"
            )

    def _wait_for_job_completion(self, job_key: str) -> Dict[str, Any]:
        """
        Wait for UiPath job to complete

        Args:
            job_key: The UiPath job key (not ID)

        Returns:
            Job result with output arguments
        """
        max_attempts = self.config.timeout_seconds
        attempt = 0

        while attempt < max_attempts:
            try:
                # Get job status using job_key as positional argument
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
                    raise IntegrationError(
                        error_msg,
                        details={
                            "service": "UiPath Processes",
                            "operation": "wait_for_job_completion",
                            "job_key": job_key,
                            "job_state": job_status.state
                        }
                    )

                # Job still running, wait and retry
                logger.debug(f"Job {job_key} state: {job_status.state}, waiting...")
                time.sleep(1)
                attempt += 1

            except Exception as e:
                logger.error(f"Error checking job status: {e}")
                raise

        # Timeout reached
        raise TimeoutError(f"Job {job_key} did not complete within {self.config.timeout_seconds} seconds")

    def _parse_job_output(self, output_args: Any, ticket_id: str) -> TicketInfo:
        """
        Parse UiPath job output into TicketInfo object using configured mapping

        Handles None values gracefully by converting them to configured defaults.

        Args:
            output_args: Output arguments from UiPath job (can be string or dict)
            ticket_id: Original ticket ID

        Returns:
            TicketInfo object

        Raises:
            ValidationError: If parsing fails or data is invalid
        """
        try:
            # Parse output_args if it's a JSON string
            if isinstance(output_args, str):
                logger.debug("Output arguments received as string, parsing JSON...")
                output_args = json.loads(output_args)

            # Log received output arguments for debugging
            logger.debug(f"Received output arguments: {list(output_args.keys())}")

            # Map output arguments using configuration
            mapped_data = {}
            for internal_field, uipath_arg in self.config.output_mapping.items():
                value = output_args.get(uipath_arg)

                # Handle None values by converting to default
                if value is None:
                    default_value = self.config.output_defaults.get(internal_field)
                    logger.warning(
                        f"Field '{internal_field}' (from UiPath arg '{uipath_arg}') is None, "
                        f"using default: '{default_value}'"
                    )
                    value = default_value

                # Log if field was missing entirely from output
                if uipath_arg not in output_args:
                    logger.warning(
                        f"Output argument '{uipath_arg}' not found in UiPath response, "
                        f"using default for '{internal_field}': '{value}'"
                    )

                mapped_data[internal_field] = value

            # Create TicketInfo object dynamically from mapped data
            ticket_data = {"ticket_id": ticket_id, **mapped_data}

            # Create TicketInfo with all fields from mapping
            ticket_info = TicketInfo(**ticket_data)

            logger.debug(f"Successfully parsed ticket info: {ticket_info.ticket_id}")
            return ticket_info

        except Exception as e:
            logger.error(f"Failed to parse job output: {e}")
            logger.error(f"Output arguments received: {output_args}")
            # Wrap in ValidationError with context
            raise ValidationError(
                f"Failed to parse ticket data from UiPath",
                details={
                    "ticket_id": ticket_id,
                    "output_args_keys": list(output_args.keys()) if isinstance(output_args, dict) else "not_dict",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )

# Convenience function for LangGraph node
async def fetch_ticket_from_uipath(ticket_id: str) -> TicketInfo:
    """
    Convenience function to fetch ticket info from UiPath

    Args:
        ticket_id: Ticket ID to retrieve

    Returns:
        TicketInfo object with ticket details
    """
    integration = UiPathIntegration()
    return await integration.get_ticket_info(ticket_id)