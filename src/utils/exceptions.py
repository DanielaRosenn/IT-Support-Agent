"""
Custom exceptions for IT Support Agent

This module defines the exception hierarchy used throughout the agent for
consistent error handling and recovery strategies.
"""

from typing import Dict, Any, Optional


class AgentException(Exception):
    """
    Base exception for all agent errors.

    All custom exceptions should inherit from this to allow catching all
    agent-specific errors with a single except clause.

    Attributes:
        message: Human-readable error description
        details: Additional context about the error (dict)
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class RecoverableError(AgentException):
    """
    Errors we can recover from with degraded functionality.

    These errors indicate a problem that doesn't require stopping execution.
    The system should log a warning, use a fallback strategy, and continue.

    Examples:
        - External API temporarily unavailable (use cache)
        - Optional data source fails (continue without that data)
        - LLM returns malformed response (retry or use default)
        - Article search fails (continue with empty results)

    Handling Strategy:
        1. Log warning with context
        2. Use fallback/default value
        3. Continue execution with degraded functionality
        4. Track degradation in metadata for reporting
    """
    pass


class FatalError(AgentException):
    """
    Errors that should stop execution immediately.

    These errors indicate a critical problem that prevents the agent from
    completing its task. Execution should stop and return an error to the user.

    Examples:
        - Invalid ticket ID format
        - Required configuration missing
        - Authentication failure
        - Critical service unavailable (no fallback possible)
        - Corrupted state that prevents further processing

    Handling Strategy:
        1. Log error with full context and stack trace
        2. Stop execution immediately
        3. Return structured error response to user
        4. Do not attempt recovery
    """
    pass


# ============================================================================
# Specific Error Types
# ============================================================================


class ValidationError(FatalError):
    """
    Input validation failures.

    Raised when user-provided input fails validation checks.

    Examples:
        - Invalid ticket ID format
        - Missing required fields
        - Invalid parameter values
    """
    pass


class IntegrationError(RecoverableError):
    """
    External integration failures.

    Raised when communication with external services fails. These are typically
    recoverable because we can retry, use cache, or continue without the data.

    Examples:
        - UiPath process invocation failure
        - UiPath Storage Bucket unavailable
        - Context Grounding service timeout
        - Network errors

    Details should include:
        - service: Name of the service (e.g., "UiPath Storage Bucket")
        - operation: What operation failed (e.g., "fetch_it_actions")
        - retry_count: Number of retries attempted (if applicable)
    """
    pass


class LLMError(RecoverableError):
    """
    LLM invocation or response parsing failures.

    Raised when LLM calls fail or return unparseable responses. These are
    recoverable because we can retry, use simpler prompts, or use fallback logic.

    Examples:
        - LLM API timeout
        - Malformed JSON response
        - Rate limit exceeded
        - Model unavailable

    Details should include:
        - operation: What LLM operation failed (e.g., "extract_keywords")
        - prompt_length: Length of prompt sent (for debugging)
        - response_preview: First 100 chars of response (if any)
    """
    pass


class ConfigurationError(FatalError):
    """
    Missing or invalid configuration.

    Raised when required configuration is missing or invalid. These are fatal
    because the agent cannot function without proper configuration.

    Examples:
        - Missing environment variables (AWS credentials, UiPath URL)
        - Invalid config values (negative thresholds, empty required fields)
        - Configuration file not found

    Details should include:
        - config_key: Name of missing/invalid config
        - expected: What was expected
        - actual: What was found (if applicable)
    """
    pass


class StateError(FatalError):
    """
    Invalid state encountered during graph execution.

    Raised when the graph state is corrupted or missing required fields that
    should have been populated by previous nodes.

    Examples:
        - Required state field missing (e.g., ticket_info not set)
        - State field has invalid type
        - State corruption detected

    Details should include:
        - node: Name of node where error occurred
        - missing_field: Name of missing state field (if applicable)
        - state_keys: Available state keys for debugging
    """
    pass


# ============================================================================
# Helper Functions
# ============================================================================


def wrap_integration_error(
    exception: Exception,
    service: str,
    operation: str,
    retry_count: int = 0
) -> IntegrationError:
    """
    Wrap a generic exception as an IntegrationError with context.

    Args:
        exception: The original exception
        service: Name of the external service
        operation: What operation was being performed
        retry_count: Number of retries attempted

    Returns:
        IntegrationError with wrapped context
    """
    return IntegrationError(
        message=f"{service} integration failed during {operation}: {str(exception)}",
        details={
            "service": service,
            "operation": operation,
            "retry_count": retry_count,
            "original_error": str(exception),
            "error_type": type(exception).__name__
        }
    )


def wrap_llm_error(
    exception: Exception,
    operation: str,
    prompt_length: int = 0,
    response_preview: str = ""
) -> LLMError:
    """
    Wrap a generic exception as an LLMError with context.

    Args:
        exception: The original exception
        operation: What LLM operation was being performed
        prompt_length: Length of prompt sent
        response_preview: Preview of response (first 100 chars)

    Returns:
        LLMError with wrapped context
    """
    return LLMError(
        message=f"LLM operation '{operation}' failed: {str(exception)}",
        details={
            "operation": operation,
            "prompt_length": prompt_length,
            "response_preview": response_preview[:100],
            "original_error": str(exception),
            "error_type": type(exception).__name__
        }
    )
