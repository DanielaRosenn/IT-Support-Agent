"""
Prompt Utilities - All LLM prompt construction and response processing

This module combines all prompt-related utilities in one place:
1. Ticket Context Building - Format ticket data for prompts
2. Prompt Construction - PromptBuilder class and convenience functions
3. Response Processing - Parse, clean, and log LLM responses

Replaces: prompt_helpers.py and prompt_builder.py
"""

import json
import logging
from typing import Any, Dict, List, Tuple, Optional
from src.config import config
from src.config.prompts import CATO_AGENT_GUIDELINES, JSON_INSTRUCTIONS


# ============================================================================
# SECTION 1: TICKET CONTEXT BUILDING
# ============================================================================


def build_ticket_context(ticket_info: Any) -> str:
    """
    Build formatted ticket context from TicketInfo object.
    Uses fields defined in config.UIPATH_OUTPUT_MAPPING dynamically.

    Args:
        ticket_info: TicketInfo object with ticket data

    Returns:
        Formatted string with all ticket fields

    Example output:
        TICKET ID: 12345
        DESCRIPTION: User needs password reset
        CATEGORY: Software
        SUBJECT: Reset Password
        REQUESTER: John Doe
        REQUESTER EMAIL: john.doe@catonetworks.com
    """
    # Always include ticket_id
    context_lines = [f"TICKET ID: {ticket_info.ticket_id}"]

    # Add all fields from UIPATH_OUTPUT_MAPPING
    for field_name in config.UIPATH_OUTPUT_MAPPING.keys():
        field_value = getattr(ticket_info, field_name, "Not available")
        # Format field name for display (convert snake_case to Title Case)
        display_name = field_name.replace("_", " ").upper()
        context_lines.append(f"{display_name}: {field_value}")

    return "\n".join(context_lines)


# ============================================================================
# SECTION 2: PROMPT CONSTRUCTION
# ============================================================================


class PromptBuilder:
    """
    Fluent API for building consistent LLM prompts.

    Eliminates duplication and ensures consistent formatting across all nodes.

    Example usage:
        prompt = (PromptBuilder("IT ACTION CLASSIFICATION")
            .add_section("TICKET TO ANALYZE", ticket_context)
            .add_section("AVAILABLE IT ACTIONS", actions_text)
            .add_section("YOUR ANALYSIS TASK", instructions)
            .build())
    """

    SEPARATOR = "═" * 79

    def __init__(self, task_title: str, include_guidelines: bool = True):
        """
        Initialize prompt builder.

        Args:
            task_title: Title for the task section (e.g., "IT ACTION CLASSIFICATION")
            include_guidelines: Whether to include CATO_AGENT_GUIDELINES at the top
        """
        self.task_title = task_title
        self.sections: List[Tuple[Optional[str], str]] = []
        self.include_guidelines = include_guidelines
        self.response_format = "json"  # or "text"

    def add_section(self, title: Optional[str], content: str) -> 'PromptBuilder':
        """
        Add a section to the prompt.

        Args:
            title: Section title (optional). If None, adds content without header.
            content: Section content

        Returns:
            Self for method chaining
        """
        self.sections.append((title, content))
        return self

    def set_response_format(self, format_type: str) -> 'PromptBuilder':
        """
        Set response format type.

        Args:
            format_type: Either "json" or "text"

        Returns:
            Self for method chaining
        """
        self.response_format = format_type
        return self

    def build(self) -> str:
        """
        Build the final prompt string.

        Returns:
            Complete formatted prompt ready for LLM
        """
        parts = []

        # Add guidelines if requested
        if self.include_guidelines:
            parts.append(CATO_AGENT_GUIDELINES)
            parts.append("")

        # Add task header
        parts.append(self.SEPARATOR)
        parts.append(f"YOUR TASK: {self.task_title}")
        parts.append(self.SEPARATOR)
        parts.append("")

        # Add all sections
        for title, content in self.sections:
            if title:
                parts.append(self.SEPARATOR)
                parts.append(title)
                parts.append(self.SEPARATOR)
                parts.append("")
            parts.append(content)
            parts.append("")

        # Add response format instructions
        if self.response_format == "json":
            parts.append(self.SEPARATOR)
            parts.append("RESPONSE FORMAT (MUST BE VALID JSON)")
            parts.append(self.SEPARATOR)
            parts.append("")
            parts.append(JSON_INSTRUCTIONS)
            parts.append("")
            parts.append("JSON Response:")

        return "\n".join(parts)


# ============================================================================
# Convenience Functions for Common Prompt Patterns
# ============================================================================


def build_classification_prompt(
    task_title: str,
    ticket_context: str,
    classification_data: str,
    instructions: str,
    examples: Optional[str] = None
) -> str:
    """
    Build a classification prompt (used by check_it_actions, extract_it_action_data, etc.).

    Args:
        task_title: Task title (e.g., "IT ACTION CLASSIFICATION")
        ticket_context: Formatted ticket information
        classification_data: Available classification data (e.g., IT actions, categories)
        instructions: Analysis instructions for the LLM
        examples: Optional examples section

    Returns:
        Complete formatted prompt
    """
    builder = (PromptBuilder(task_title)
        .add_section("TICKET TO ANALYZE", ticket_context)
        .add_section(None, classification_data)  # No title, data has its own header
        .add_section("YOUR ANALYSIS TASK", instructions))

    if examples:
        builder.add_section("EXAMPLES", examples)

    return builder.build()


def build_analysis_prompt(
    task_title: str,
    ticket_context: str,
    available_info: str,
    analysis_instructions: str,
    response_format: str = "json"
) -> str:
    """
    Build an analysis prompt (used by knowledge_search, evaluate_sufficiency, etc.).

    Args:
        task_title: Task title (e.g., "KNOWLEDGE SUFFICIENCY ANALYSIS")
        ticket_context: Formatted ticket information
        available_info: Available information to analyze (e.g., knowledge sources)
        analysis_instructions: Instructions for the analysis
        response_format: "json" or "text"

    Returns:
        Complete formatted prompt
    """
    return (PromptBuilder(task_title)
        .add_section("TICKET INFORMATION", ticket_context)
        .add_section("AVAILABLE KNOWLEDGE SOURCES", available_info)
        .add_section("YOUR ANALYSIS TASK", analysis_instructions)
        .set_response_format(response_format)
        .build())


def build_response_generation_prompt(
    task_title: str,
    ticket_context: str,
    available_info: str,
    response_instructions: str,
    response_style_guide: Optional[str] = None,
    include_json: bool = False
) -> str:
    """
    Build a response generation prompt (used by generate_ticket_response).

    Args:
        task_title: Task title (e.g., "GENERATE TICKET RESPONSE")
        ticket_context: Formatted ticket information
        available_info: Available information for response (e.g., KB articles, solutions)
        response_instructions: Instructions for generating the response
        response_style_guide: Optional style guide for responses
        include_json: Whether response should be JSON or plain text

    Returns:
        Complete formatted prompt
    """
    builder = (PromptBuilder(task_title)
        .add_section("TICKET CONTEXT", ticket_context)
        .add_section("COLLECTED INFORMATION FROM ALL SOURCES", available_info)
        .add_section("YOUR RESPONSE TASK", response_instructions))

    if response_style_guide:
        builder.add_section("RESPONSE STYLE (CRITICAL - FOLLOW EXACTLY)", response_style_guide)

    if not include_json:
        builder.set_response_format("text")

    return builder.build()


def build_extraction_prompt(
    task_title: str,
    ticket_context: str,
    extraction_schema: str,
    extraction_instructions: str
) -> str:
    """
    Build an extraction prompt (used by extract_it_action_data, extract_web_search_topics).

    Args:
        task_title: Task title (e.g., "EXTRACT IT ACTION DATA")
        ticket_context: Formatted ticket information
        extraction_schema: Schema/structure of what to extract
        extraction_instructions: Instructions for extraction

    Returns:
        Complete formatted prompt
    """
    return (PromptBuilder(task_title)
        .add_section("TICKET TO ANALYZE", ticket_context)
        .add_section("EXTRACTION SCHEMA", extraction_schema)
        .add_section("EXTRACTION INSTRUCTIONS", extraction_instructions)
        .build())


# ============================================================================
# SECTION 3: RESPONSE PROCESSING
# ============================================================================


def clean_llm_json_response(response: str) -> str:
    """
    Clean LLM response by removing markdown code blocks.

    Args:
        response: Raw LLM response that may contain ```json markers

    Returns:
        Cleaned JSON string ready for parsing
    """
    response_clean = response.strip()

    # Remove ```json prefix
    if response_clean.startswith("```json"):
        response_clean = response_clean.split("```json", 1)[1]
    # Remove ``` prefix
    elif response_clean.startswith("```"):
        response_clean = response_clean.split("```", 1)[1]

    # Remove ``` suffix
    if response_clean.endswith("```"):
        response_clean = response_clean.rsplit("```", 1)[0]

    return response_clean.strip()


def log_llm_response(logger: logging.Logger, response: str, title: str) -> None:
    """
    Log LLM response with visual separators.

    Args:
        logger: Logger instance to use
        response: LLM response text to log
        title: Title for the response (e.g., "IT ACTION MATCH RESPONSE")
    """
    logger.info("=" * 80)
    logger.info(f"LLM {title}:")
    logger.info("=" * 80)
    logger.info(response)
    logger.info("=" * 80)


def parse_llm_json_response(
    response: str,
    logger: logging.Logger,
    error_context: str = "LLM response"
) -> Dict[str, Any]:
    """
    Parse and validate LLM JSON response with error handling.

    Args:
        response: Raw LLM response
        logger: Logger instance for error reporting
        error_context: Context description for error messages

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If parsing fails
        ValueError: If validation fails
    """
    try:
        cleaned = clean_llm_json_response(response)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse {error_context}: {e}")
        logger.error(f"Raw response: {response}")
        raise
