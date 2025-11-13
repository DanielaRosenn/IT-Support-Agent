"""
Step 4: Extract IT Action Data

When an IT action matches, extract the required information from the ticket.
If information is missing, generate questions to collect it from the user.
"""

import logging
import json
from typing import Dict, Any, List
from src.graph.state import GraphState
from src.utils.llm_service import get_llm_service
from src.utils.prompt_utils import (
    build_ticket_context,
    log_llm_response,
    parse_llm_json_response
)
from src.config.prompts import (
    CATO_AGENT_GUIDELINES,
    JSON_INSTRUCTIONS,
    EXTRACTION_SYSTEM_MESSAGE,
    CLARIFYING_QUESTIONS_SYSTEM_MESSAGE
)

logger = logging.getLogger(__name__)


async def extract_it_action_data(state: GraphState) -> dict:
    """
    Step 4: Extract IT Action Required Data

    When an IT action matches (Step 3), this node:
    - Extracts required information from the ticket based on ticket_information_required fields
    - Identifies any missing required information
    - Generates clarifying questions for missing data

    Args:
        state: Current graph state containing ticket_info and it_action_match

    Returns:
        dict: Updated state with:
            - ticket_information: Dict of extracted values for each required field
            - missing_information: List of field names that are missing
            - clarifying_questions: List of questions to ask user (if any missing)
            - can_proceed_with_it_action: Boolean (True if all information present, False if questions needed)
    """
    ticket_info = state["ticket_info"]
    it_action_match = state.get("it_action_match", {})

    if not it_action_match.get("is_match"):
        logger.warning("[Step 4] Called extract_it_action_data but no IT action match found")
        return {
            "ticket_information": {},
            "missing_information": [],
            "clarifying_questions": [],
            "can_proceed_with_it_action": False
        }

    matched_action_name = it_action_match.get("matched_action_name")
    ticket_information_required = it_action_match.get("ticket_information_required", {})

    logger.info(f"[Step 4] Extracting IT Action Data")
    logger.info(f"  - Matched Action: {matched_action_name}")
    logger.info(f"  - Required Fields: {list(ticket_information_required.keys())}")

    try:
        # Use LLM to extract required information from ticket
        extraction_result = await _extract_required_information(
            ticket_info=ticket_info,
            matched_action_name=matched_action_name,
            ticket_information_required=ticket_information_required
        )

        extracted_data = extraction_result["extracted_data"]
        missing_data = extraction_result["missing_data"]

        logger.info(f"[Step 4] Extraction Results:")
        logger.info(f"  - Extracted Fields: {list(extracted_data.keys())}")
        logger.info(f"  - Missing Fields: {missing_data}")

        # If data is missing, generate clarifying questions
        clarifying_questions = []
        can_proceed = len(missing_data) == 0

        if missing_data:
            logger.info(f"[Step 4] Generating clarifying questions for missing data...")
            clarifying_questions = await _generate_clarifying_questions(
                ticket_info=ticket_info,
                matched_action_name=matched_action_name,
                ticket_information_required=ticket_information_required,
                extracted_data=extracted_data,
                missing_data=missing_data
            )

            logger.info(f"[Step 4] Generated {len(clarifying_questions)} clarifying questions:")
            for i, question in enumerate(clarifying_questions, 1):
                logger.info(f"  [{i}] {question}")

        return {
            "ticket_information": extracted_data,
            "missing_information": missing_data,
            "clarifying_questions": clarifying_questions,
            "can_proceed_with_it_action": can_proceed
        }

    except Exception as e:
        logger.error(f"[Step 4] Failed to extract IT action data: {str(e)}")
        return {
            "ticket_information": {},
            "missing_information": list(ticket_information_required.keys()),
            "clarifying_questions": [f"Unable to process request. Please provide: {', '.join(ticket_information_required.keys())}"],
            "can_proceed_with_it_action": False
        }


async def _extract_required_information(
    ticket_info: Any,
    matched_action_name: str,
    ticket_information_required: Dict[str, str]
) -> Dict[str, Any]:
    """
    Use LLM to extract required information fields from ticket.

    Args:
        ticket_info: TicketInfo object
        matched_action_name: Name of matched IT action
        ticket_information_required: Dict of {field_name: field_description}

    Returns:
        Dict with:
            - extracted_data: {field_name: extracted_value or None}
            - missing_data: [field_names that are missing]
    """
    llm_service = get_llm_service()

    # Build ticket context dynamically from config
    ticket_context = build_ticket_context(ticket_info)

    # Format required fields for prompt with enhanced format handling
    required_fields_text = []
    for field_name, field_spec in ticket_information_required.items():
        # Handle both simple string and complex dict formats
        if isinstance(field_spec, dict):
            # Extended format with schema
            description = field_spec.get("description", field_name)
            format_type = field_spec.get("format", "string")
            schema = field_spec.get("schema", {})
            example = field_spec.get("example", "")

            field_text = f"- **{field_name}** ({format_type}):\n  Description: {description}"

            if schema:
                if format_type == "array_of_objects" and "items" in schema:
                    field_text += "\n  Required structure:"
                    for key, desc in schema["items"].items():
                        field_text += f"\n    - {key}: {desc}"

            if example:
                field_text += f"\n  Example: {example}"

            required_fields_text.append(field_text)
        else:
            # Simple string format
            required_fields_text.append(f"- **{field_name}**: {field_spec}")

    required_fields_text = "\n".join(required_fields_text)

    prompt = f"""{CATO_AGENT_GUIDELINES}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK: EXTRACT IT ACTION REQUIRED INFORMATION
═══════════════════════════════════════════════════════════════════════════════

Extract required information from this IT support ticket for the matched IT action.

═══════════════════════════════════════════════════════════════════════════════
TICKET INFORMATION
═══════════════════════════════════════════════════════════════════════════════

{ticket_context}

═══════════════════════════════════════════════════════════════════════════════
MATCHED IT ACTION
═══════════════════════════════════════════════════════════════════════════════

ACTION NAME: {matched_action_name}

REQUIRED INFORMATION TO EXTRACT:
{required_fields_text}

═══════════════════════════════════════════════════════════════════════════════
YOUR EXTRACTION TASK
═══════════════════════════════════════════════════════════════════════════════

For EACH required field above:
1. Search the ticket (subject, description, requester info) for this information
2. If found: Extract the exact value
3. If not found or unclear: Mark as null

EXTRACTION RULES:
- Extract ONLY what is explicitly stated in the ticket
- Do NOT infer, assume, or make up information
- If a field has multiple values (e.g., list of apps), extract all of them
- If a field is mentioned but value is unclear, mark as null
- Be precise: extract exact names, emails, system names as stated
- **CRITICAL**: If a field specifies a format (e.g., array_of_objects with schema), you MUST return data in that exact JSON format
- For structured fields, return valid JSON arrays or objects that match the specified schema exactly

═══════════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT (MUST BE VALID JSON)
═══════════════════════════════════════════════════════════════════════════════

{JSON_INSTRUCTIONS}

Respond with ONLY valid JSON:

{{
    "extracted_data": {{
        "field_name_1": "extracted value" or null or [array] or {{"object"}},
        "field_name_2": "extracted value" or null,
        ...
    }},
    "extraction_notes": "Brief explanation of what was found and what's missing"
}}

EXAMPLES:

Example 1 - Simple string field:
If required field is:
- target_user_email: Email of user who needs the action performed

Ticket: "Please grant John Doe (john.doe@company.com) access"

Response:
{{
    "extracted_data": {{
        "target_user_email": "john.doe@company.com"
    }},
    "extraction_notes": "Found target user email"
}}

Example 2 - Structured array_of_objects field:
If required field is:
- list_apps_permissions (array_of_objects):
  Required structure:
    - app: Application name
    - email: User email address
    - business_justification: Business reason
  Example: [{{"app":"Jira","email":"user@company.com","business_justification":"project work"}}]

Ticket: "Need Gong license for gaya.granot@catonetworks.com for daily work, and PyCharm for daniela.rosenstein@catonetworks.com"

Response:
{{
    "extracted_data": {{
        "list_apps_permissions": [
            {{"app": "Gong", "email": "gaya.granot@catonetworks.com", "business_justification": "for daily work"}},
            {{"app": "PyCharm", "email": "daniela.rosenstein@catonetworks.com", "business_justification": ""}}
        ]
    }},
    "extraction_notes": "Found 2 application requests with user emails and partial business justification"
}}

JSON Response:"""

    logger.debug("Sending information extraction request to LLM")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=600,
        temperature=0.0,  # Zero temperature for precise extraction
        system_message=EXTRACTION_SYSTEM_MESSAGE
    )

    # Log and parse response
    log_llm_response(logger, response, "EXTRACTION RESPONSE")

    try:
        result = parse_llm_json_response(response, logger, "extraction")

        extracted_data = result.get("extracted_data", {})
        extraction_notes = result.get("extraction_notes", "")

        logger.info(f"[Step 4] Extraction Notes: {extraction_notes}")

        # Identify missing data (fields with null values OR empty strings/arrays)
        # Also validate nested fields in array_of_objects structures
        missing_data = []
        for field_name in ticket_information_required.keys():
            value = extracted_data.get(field_name)
            field_spec = ticket_information_required[field_name]

            # Check if value is truly missing or empty (top-level check)
            is_missing = (
                value is None or  # Explicitly null
                (isinstance(value, str) and value.strip() == "") or  # Empty string
                (isinstance(value, list) and len(value) == 0) or  # Empty array
                (isinstance(value, dict) and len(value) == 0)  # Empty dict
            )

            if is_missing:
                missing_data.append(field_name)
                continue

            # For array_of_objects fields, validate nested required fields
            if isinstance(field_spec, dict) and field_spec.get("format") == "array_of_objects":
                if isinstance(value, list) and len(value) > 0:
                    schema = field_spec.get("schema", {})

                    # Determine which nested fields to validate
                    # If schema.required is specified, use it
                    # Otherwise, assume ALL fields in schema.items are required (except metadata fields)
                    required_nested_fields = schema.get("required", [])

                    if not required_nested_fields and "items" in schema:
                        # Default: all fields in schema.items are required
                        # (This handles the common case where schema.required is not specified)
                        required_nested_fields = list(schema["items"].keys())

                    # Specific rule: for list_apps_permissions, ALWAYS require business_justification
                    # This is a critical field that should never be empty
                    if field_name == "list_apps_permissions" and "business_justification" not in required_nested_fields:
                        if "items" in schema and "business_justification" in schema["items"]:
                            required_nested_fields.append("business_justification")

                    # Check if any nested required fields are empty across all array items
                    has_empty_nested_fields = False
                    empty_fields_info = []  # Track which fields are empty for logging

                    for idx, item in enumerate(value):
                        if isinstance(item, dict):
                            for nested_field in required_nested_fields:
                                nested_value = item.get(nested_field)
                                if (nested_value is None or
                                    (isinstance(nested_value, str) and nested_value.strip() == "")):
                                    has_empty_nested_fields = True
                                    empty_fields_info.append(f"Item {idx}: '{nested_field}' is empty")
                                    break
                        if has_empty_nested_fields:
                            break

                    if has_empty_nested_fields:
                        missing_data.append(field_name)
                        logger.info(f"[Step 4] Nested field validation failed for '{field_name}': {', '.join(empty_fields_info)}")

        return {
            "extracted_data": extracted_data,
            "missing_data": missing_data
        }

    except (json.JSONDecodeError, ValueError):
        # Error already logged by parse_llm_json_response
        # Return all fields as missing
        return {
            "extracted_data": {field: None for field in ticket_information_required.keys()},
            "missing_data": list(ticket_information_required.keys())
        }


async def _generate_clarifying_questions(
    ticket_info: Any,
    matched_action_name: str,
    ticket_information_required: Dict[str, str],
    extracted_data: Dict[str, Any],
    missing_data: List[str]
) -> List[str]:
    """
    Generate clarifying questions for missing required information.

    Args:
        ticket_info: TicketInfo object
        matched_action_name: Name of matched IT action
        ticket_information_required: Dict of {field_name: field_description}
        extracted_data: Dict of successfully extracted values
        missing_data: List of field names that are missing

    Returns:
        List of clarifying questions to ask the user
    """
    llm_service = get_llm_service()

    # Format what we have vs what's missing
    # For array_of_objects fields, show the structure with empty nested fields highlighted
    extracted_fields_text_parts = []
    for field, value in extracted_data.items():
        if value is None:
            continue

        field_spec = ticket_information_required.get(field)

        # Special handling for array_of_objects to show which nested fields are empty
        if isinstance(field_spec, dict) and field_spec.get("format") == "array_of_objects":
            if isinstance(value, list) and len(value) > 0:
                schema = field_spec.get("schema", {})
                required_nested_fields = schema.get("required", [])

                if not required_nested_fields and "items" in schema:
                    required_nested_fields = list(schema["items"].keys())

                if field == "list_apps_permissions" and "business_justification" not in required_nested_fields:
                    if "items" in schema and "business_justification" in schema["items"]:
                        required_nested_fields.append("business_justification")

                # Show array structure with empty fields marked
                array_text = f"- **{field}** (array with {len(value)} item(s)):\n"
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        array_text += f"  Item {idx+1}:\n"
                        for key, val in item.items():
                            is_empty = val is None or (isinstance(val, str) and val.strip() == "")
                            if is_empty and key in required_nested_fields:
                                array_text += f"    - {key}: [EMPTY - NEEDS VALUE]\n"
                            else:
                                array_text += f"    - {key}: {val}\n"
                extracted_fields_text_parts.append(array_text)
            else:
                extracted_fields_text_parts.append(f"- **{field}**: {value}")
        else:
            extracted_fields_text_parts.append(f"- **{field}**: {value}")

    extracted_fields_text = "\n".join(extracted_fields_text_parts) if extracted_fields_text_parts else "None"

    # Format missing fields with context about what's missing
    missing_fields_text_parts = []
    for field in missing_data:
        field_spec = ticket_information_required[field]

        # For array_of_objects, explain which nested field is missing
        if isinstance(field_spec, dict) and field_spec.get("format") == "array_of_objects":
            if field in extracted_data and isinstance(extracted_data[field], list):
                # Array exists but has empty nested fields
                missing_fields_text_parts.append(
                    f"- **{field}**: Has empty required nested fields (see extracted data above for details)"
                )
            else:
                # Array itself is missing
                missing_fields_text_parts.append(
                    f"- **{field}**: {field_spec.get('description', field_spec)}"
                )
        else:
            missing_fields_text_parts.append(
                f"- **{field}**: {field_spec if isinstance(field_spec, str) else field_spec.get('description', field)}"
            )

    missing_fields_text = "\n".join(missing_fields_text_parts)

    prompt = f"""{CATO_AGENT_GUIDELINES}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK: GENERATE CLARIFYING QUESTIONS
═══════════════════════════════════════════════════════════════════════════════

Generate clarifying questions to collect missing information from the user.

═══════════════════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════════════════

TICKET ID: {ticket_info.ticket_id}
SUBJECT: {ticket_info.subject}
MATCHED IT ACTION: {matched_action_name}

INFORMATION ALREADY EXTRACTED:
{extracted_fields_text}

MISSING INFORMATION NEEDED:
{missing_fields_text}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Generate CONCISE, PROFESSIONAL questions to ask the user for the missing information.

QUESTION WRITING RULES:
1. One question per missing field
2. Be specific about what you need
3. Use professional IT support tone (not overly formal)
4. Reference the ticket context where helpful
5. If multiple related fields are missing, you can combine into one question
6. Keep questions SHORT (1-2 sentences max)

EXAMPLES:

Missing: target_user_email
Good: "Which user should receive this access? Please provide their email address."
Bad: "Can you please kindly provide us with the email address of the target user?"

Missing: list_apps_permissions
Good: "Which applications or permissions do you need?"
Bad: "We need to know what applications you are requesting access to at this time."

Missing: business_justification
Good: "What's the business reason for this request?"
Bad: "Could you elaborate on the business justification for this access request?"

═══════════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT (MUST BE VALID JSON)
═══════════════════════════════════════════════════════════════════════════════

{JSON_INSTRUCTIONS}

Respond with ONLY valid JSON:

{{
    "questions": [
        "Question 1 for missing field?",
        "Question 2 for missing field?"
    ]
}}

JSON Response:"""

    logger.debug("Sending clarifying questions generation request to LLM")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=400,
        temperature=0.3,  # Slightly creative for natural questions
        system_message=CLARIFYING_QUESTIONS_SYSTEM_MESSAGE
    )

    # Log and parse response
    log_llm_response(logger, response, "CLARIFYING QUESTIONS RESPONSE")

    try:
        result = parse_llm_json_response(response, logger, "clarifying questions")
        questions = result.get("questions", [])

        if not questions:
            # Fallback: generate basic questions
            questions = [
                f"Please provide: {ticket_information_required[field]}"
                for field in missing_data
            ]

        return questions

    except (json.JSONDecodeError, ValueError):
        # Error already logged by parse_llm_json_response
        # Fallback: generate basic questions
        return [
            f"Please provide: {ticket_information_required[field]}"
            for field in missing_data
        ]