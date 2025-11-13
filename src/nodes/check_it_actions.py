"""
Step 3: Check IT Actions Match

Determines if the ticket matches any predefined IT action categories.
CRITICAL: Misclassification can route tickets incorrectly - use high confidence threshold.
"""

import logging
import json
from typing import Dict, Any
from src.graph.state import GraphState
from src.integrations.uipath_storage_bucket import fetch_it_actions_from_bucket
from src.utils.llm_service import get_llm_service
from src.utils.prompt_utils import (
    build_ticket_context,
    log_llm_response,
    parse_llm_json_response,
    build_classification_prompt
)
from src.config.prompts import JSON_INSTRUCTIONS
from src.config import config
from src.utils.exceptions import RecoverableError, LLMError, IntegrationError

logger = logging.getLogger(__name__)


async def check_it_actions(state: GraphState) -> dict:
    """
    Step 3: Check if ticket matches any IT action category.

    IMPORTANT: IT actions represent specific, well-defined procedures that
    require IT team intervention. We must be very confident before routing
    to this path to avoid false positives.

    Args:
        state: Current graph state containing ticket_info

    Returns:
        dict: Updated state with it_action_match and is_it_action_match
    """
    ticket_info = state["ticket_info"]

    logger.info(f"[Step 3] Checking IT Actions match for ticket")
    logger.info(f"  - Category: {ticket_info.category}")
    logger.info(f"  - Subject: {ticket_info.subject}")

    try:
        # [Step 3a] Load IT Actions from Storage Bucket
        logger.info("[Step 3a] Loading IT Actions from Storage Bucket...")

        try:
            it_actions_data = fetch_it_actions_from_bucket()
            it_human_actions = it_actions_data.get("it_human_actions", {})
        except Exception as e:
            # Storage bucket failure is recoverable - continue without IT actions
            raise IntegrationError(
                "Failed to load IT actions from Storage Bucket",
                details={
                    "service": "UiPath Storage Bucket",
                    "operation": "fetch_it_actions",
                    "error": str(e)
                }
            )

        if not it_human_actions:
            logger.warning("[Step 3a] No IT actions found in Storage Bucket")
            return {
                "it_action_match": {
                    "is_match": False,
                    "matched_action_name": None,
                    "confidence": 0.0,
                    "reasoning": "No IT actions available in configuration"
                },
                "is_it_action_match": False
            }

        logger.info(f"[Step 3a] Loaded {len(it_human_actions)} IT action categories")

        # [Step 3b] Use LLM to analyze ticket against IT actions
        logger.info("[Step 3b] Analyzing ticket against IT actions using LLM...")
        logger.info(f"[Step 3b] Match threshold set to: {config.IT_ACTION_MATCH_THRESHOLD}")

        match_result = await _analyze_it_action_match(
            ticket_info=ticket_info,
            it_human_actions=it_human_actions
        )

        # Apply confidence threshold
        is_match = match_result["is_match"] and match_result["confidence"] >= config.IT_ACTION_MATCH_THRESHOLD

        if match_result["is_match"] and not is_match:
            logger.warning(
                f"[Step 3] LLM indicated match but confidence {match_result['confidence']:.2f} "
                f"is below threshold {config.IT_ACTION_MATCH_THRESHOLD}. Treating as NO MATCH."
            )
            match_result["is_match"] = False
            match_result["reasoning"] += f" (Confidence {match_result['confidence']:.2f} below required {config.IT_ACTION_MATCH_THRESHOLD})"

        logger.info(f"[Step 3] IT Action Match Result:")
        logger.info(f"  - Is Match: {is_match}")
        logger.info(f"  - Confidence: {match_result['confidence']:.2f}")
        if is_match:
            logger.info(f"  - Matched Action: {match_result['matched_action_name']}")
            logger.info(f"  - Ticket Information Required: {list(match_result.get('ticket_information_required', {}).keys())}")
        logger.info(f"  - Reasoning: {match_result['reasoning']}")

        return {
            "it_action_match": match_result,
            "is_it_action_match": is_match
        }

    except (IntegrationError, LLMError) as e:
        # Recoverable errors - log warning and continue with no match (safe default)
        logger.warning(
            f"[Step 3] Recoverable error in IT action check: {e.message}",
            extra={"details": e.details, "ticket_id": ticket_info.ticket_id}
        )
        return {
            "it_action_match": {
                "is_match": False,
                "matched_action_name": None,
                "confidence": 0.0,
                "reasoning": f"IT action check unavailable: {e.message}"
            },
            "is_it_action_match": False
        }

    except Exception as e:
        # Unexpected error - log and continue with safe default
        logger.error(
            f"[Step 3] Unexpected error in IT action check: {str(e)}",
            exc_info=True,
            extra={"ticket_id": ticket_info.ticket_id}
        )
        return {
            "it_action_match": {
                "is_match": False,
                "matched_action_name": None,
                "confidence": 0.0,
                "reasoning": f"Unexpected error: {str(e)}"
            },
            "is_it_action_match": False
        }


async def _analyze_it_action_match(
    ticket_info: Any,
    it_human_actions: dict
) -> Dict[str, Any]:
    """
    Use LLM to determine if ticket matches any IT action.

    The IT actions JSON serves as INSTRUCTIONS for matching - each action
    defines specific criteria that must be met.

    Args:
        ticket_info: TicketInfo object
        it_human_actions: Dict of IT actions from it_actions.json (format: {action_name: action_config})

    Returns:
        Dict with is_match, matched_action_name, confidence, reasoning, ticket_information_required
    """
    llm_service = get_llm_service()

    # Format IT actions as detailed matching instructions
    actions_formatted = []
    for action_name, action in it_human_actions.items():
        # Extract ticket information required fields
        required_fields = action.get('ticket_information_required', {})
        required_fields_list = list(required_fields.keys())

        action_str = f"""
ACTION NAME: {action_name}
DISPLAY NAME: {action.get('name', action_name)}
CATEGORY: {action.get('category')}
DESCRIPTION: {action.get('description')}
KEYWORDS: {', '.join(action.get('keywords', []))}
LLM ANALYSIS CONTEXT: {action.get('llm_analysis_context', 'No special context')}
TICKET INFORMATION REQUIRED FROM TICKET: {', '.join(required_fields_list)}

MATCHING INSTRUCTIONS:
{action.get('llm_analysis_context', 'Match if keywords and context align')}

CONFIDENCE INDICATORS:
{json.dumps(action.get('decision_criteria', {}), indent=2)}
"""
        actions_formatted.append(action_str.strip())

    actions_text = "\n\n" + "="*80 + "\n\n".join(actions_formatted)

    # Build ticket context dynamically from config
    ticket_context = build_ticket_context(ticket_info)

    # Build classification instructions
    classification_instructions = f"""You are an expert IT ticket classifier. Your job is to determine if a support ticket matches ANY of the predefined IT action procedures below.

⚠️  CRITICAL INSTRUCTIONS ⚠️
1. IT Actions are SPECIFIC, WELL-DEFINED procedures with precise matching criteria
2. Each action has an "LLM ANALYSIS CONTEXT" - this is YOUR INSTRUCTION for when to select it
3. Only return is_match=true if you are HIGHLY CONFIDENT (confidence >= 0.90)
4. If you have ANY doubt, return is_match=false
5. False positives are MORE HARMFUL than false negatives
6. Read the LLM_ANALYSIS_CONTEXT for each action - it tells you EXACTLY when to match

For EACH IT action above:
1. **Read the LLM_ANALYSIS_CONTEXT** - this is your PRIMARY instruction
2. **Check keywords** - Are key terms present in ticket?
3. **Verify intent** - Does ticket's purpose match the action's purpose?
4. **Check required information** - Does ticket contain fields from TICKET_INFORMATION_REQUIRED?

MATCHING RULES:
✓ Return is_match=true ONLY if:
  - LLM_ANALYSIS_CONTEXT criteria are met
  - Keywords from the action appear in ticket
  - Ticket intent aligns with action's purpose
  - Confidence >= 0.90 (90% certain)
  - Only ONE action matches (if multiple, return the best match)

✗ Return is_match=false if:
  - LLM_ANALYSIS_CONTEXT criteria NOT met
  - Keywords match but intent is different
  - Ticket is vague or ambiguous
  - You're not 90%+ confident
  - Multiple actions could match (ambiguous)

Respond with ONLY valid JSON in this exact format:

{{
    "is_match": true or false,
    "matched_action_name": "action_name_from_json" or null,
    "confidence": 0.0 to 1.0 (must be >= 0.90 for is_match=true),
    "reasoning": "Detailed explanation referencing the LLM_ANALYSIS_CONTEXT, which keywords were found, how ticket aligns with action criteria, and what required information is available in the ticket.",
    "ticket_information_required": {{}} or {{"field_name": "description from json"}}
}}

IMPORTANT: If is_match=true, you MUST include "ticket_information_required" with the exact field names and descriptions from the matched action's TICKET_INFORMATION_REQUIRED section.

EXAMPLES OF GOOD REASONING:

MATCH Example:
{{
    "is_match": true,
    "matched_action_name": "application_request",
    "confidence": 0.95,
    "reasoning": "LLM_ANALYSIS_CONTEXT states 'Select this action for any application request for access or a license'. Ticket mentions 'need license for Jira' which is in keywords list. Requester email provided (john@company.com). Business justification present ('for project management'). All criteria met.",
    "ticket_information_required": {{"list_apps_permissions": "Extract application access requests..."}}
}}

NO MATCH Example:
{{
    "is_match": false,
    "matched_action_name": null,
    "confidence": 0.40,
    "reasoning": "While ticket mentions 'application', the LLM_ANALYSIS_CONTEXT for 'application_request' states it's for LICENSE requests only, not technical issues. This ticket describes a technical error with an existing app, not a license request. Intent mismatch.",
    "ticket_information_required": {{}}
}}"""

    # Build the prompt using the builder
    prompt = build_classification_prompt(
        task_title="IT ACTION CLASSIFICATION",
        ticket_context=ticket_context,
        classification_data=f"AVAILABLE IT ACTION PROCEDURES (Use LLM_ANALYSIS_CONTEXT as primary guide)\n{actions_text}",
        instructions=classification_instructions
    )

    logger.debug("Sending IT action match analysis request to LLM (strict mode)")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=800,
        temperature=0.0  # Zero temperature for maximum consistency
    )

    # Log and parse response
    log_llm_response(logger, response, "IT ACTION MATCH RESPONSE")

    try:
        result = parse_llm_json_response(response, logger, "IT action match analysis")

        # Validate required fields
        required_fields = ["is_match", "confidence", "reasoning"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Ensure matched_action_name is None if no match
        if not result["is_match"]:
            result["matched_action_name"] = None
            result["ticket_information_required"] = {}

        # Validate ticket_information_required is present for matches
        if result["is_match"] and "ticket_information_required" not in result:
            logger.warning("Match returned without ticket_information_required, adding empty dict")
            result["ticket_information_required"] = {}

        # Additional safety check: if confidence < threshold, force no match
        if result["is_match"] and result["confidence"] < config.IT_ACTION_MATCH_THRESHOLD:
            logger.warning(f"LLM returned match with low confidence {result['confidence']:.2f}, forcing no match")
            result["is_match"] = False
            result["matched_action_name"] = None
            result["ticket_information_required"] = {}
            result["reasoning"] += f" [Auto-rejected: confidence {result['confidence']:.2f} < threshold {config.IT_ACTION_MATCH_THRESHOLD}]"

        return result

    except (json.JSONDecodeError, ValueError) as e:
        # Error already logged by parse_llm_json_response
        # Return default NO-MATCH result (safe default)
        return {
            "is_match": False,
            "matched_action_name": None,
            "confidence": 0.0,
            "reasoning": f"Failed to parse LLM analysis: {str(e)}. Defaulting to NO MATCH for safety.",
            "ticket_information_required": {}
        }