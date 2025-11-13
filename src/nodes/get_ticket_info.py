"""
Step 1: Get Ticket Information

Retrieves ticket data from FreshService via UiPath process.
"""

import logging
from src.graph.state import GraphState
from src.integrations.uipath_get_job_data import fetch_ticket_from_uipath
from src.utils.exceptions import ValidationError, IntegrationError, FatalError

logger = logging.getLogger(__name__)


async def get_ticket_info(state: GraphState) -> dict:
    """
    Step 1: Get ticket information from UiPath process.

    Args:
        state: Current graph state containing ticket_id

    Returns:
        dict: Updated state with ticket_info populated

    Raises:
        FatalError: If ticket retrieval fails (stops execution)
    """
    ticket_id = state["ticket_id"]

    logger.info(f"[Step 1] Getting ticket information for ticket_id: {ticket_id}")

    try:
        # Fetch ticket from UiPath process
        # This uses the existing integration that:
        # 1. Invokes UiPath process 'getFreshTicket2.0'
        # 2. Polls for job completion
        # 3. Parses output using config.UIPATH_OUTPUT_MAPPING
        # 4. Returns TicketInfo model
        ticket_info = await fetch_ticket_from_uipath(ticket_id)

        logger.info(f"[Step 1] Successfully retrieved ticket info:")
        logger.info(f"  - Category: {ticket_info.category}")
        logger.info(f"  - Subject: {ticket_info.subject}")
        logger.info(f"  - Requester: {ticket_info.requester}")

        return {
            "ticket_info": ticket_info
        }

    except ValidationError as e:
        # Data validation failed (e.g., unable to parse UiPath output)
        logger.error(
            f"[Step 1] Ticket data validation failed: {e.message}",
            extra={"details": e.details, "ticket_id": ticket_id}
        )
        raise FatalError(
            f"Failed to validate ticket data for {ticket_id}",
            details={"original_error": e.message, "ticket_id": ticket_id}
        )

    except IntegrationError as e:
        # UiPath integration failed
        logger.error(
            f"[Step 1] UiPath integration failed: {e.message}",
            extra={"details": e.details, "ticket_id": ticket_id}
        )
        raise FatalError(
            f"Failed to retrieve ticket {ticket_id} from UiPath",
            details={"original_error": e.message, "ticket_id": ticket_id}
        )

    except Exception as e:
        # Unexpected error
        logger.error(
            f"[Step 1] Unexpected error retrieving ticket: {str(e)}",
            exc_info=True,
            extra={"ticket_id": ticket_id}
        )
        raise FatalError(
            f"Unexpected error retrieving ticket {ticket_id}: {str(e)}",
            details={"ticket_id": ticket_id}
        )