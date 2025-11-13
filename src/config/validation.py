"""
Configuration Validation Module

Validates configuration on startup to catch errors early.
Does NOT modify any existing logic.

Usage:
    # Standalone validation
    python -m src.config.validation

    # In application code
    from src.config.validation import validate_config
    validate_config()  # Raises ConfigurationError if invalid
"""

import os
import logging
from typing import List, Tuple
from src.config import config
from src.utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def validate_config() -> None:
    """
    Validate configuration values on startup.

    Performs the following checks:
    1. Required environment variables are set
    2. Thresholds are in valid ranges (0.0-1.0)
    3. Integer values are positive
    4. Temperature is in valid range
    5. Model ID is not empty

    Raises:
        ConfigurationError: If configuration is invalid

    Returns:
        None if validation passes
    """
    errors: List[str] = []

    # ========== Check Required Environment Variables ==========
    required_env = [
        "UIPATH_URL",
        "UIPATH_ACCESS_TOKEN",
        "UIPATH_ORGANIZATION_ID",
        "UIPATH_TENANT_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY"
    ]

    for var in required_env:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")

    # ========== Validate Threshold Ranges (0.0 to 1.0) ==========
    thresholds: List[Tuple[str, float]] = [
        ("KNOWLEDGE_SUFFICIENCY_THRESHOLD", config.KNOWLEDGE_SUFFICIENCY_THRESHOLD),
        ("ARTICLE_RERANK_MIN_SCORE", config.ARTICLE_RERANK_MIN_SCORE),
        ("CONFLUENCE_MEMORY_MIN_SCORE", config.CONFLUENCE_MEMORY_MIN_SCORE),
        ("IT_ACTION_MATCH_THRESHOLD", config.IT_ACTION_MATCH_THRESHOLD),
        ("AUGMENTATION_QUALITY_THRESHOLD", config.AUGMENTATION_QUALITY_THRESHOLD),
        ("KNOWLEDGE_SEARCH_MIN_CONFIDENCE", config.KNOWLEDGE_SEARCH_MIN_CONFIDENCE),
        ("CLASSIFICATION_HIGH_CONFIDENCE", config.CLASSIFICATION_HIGH_CONFIDENCE),
        ("CLASSIFICATION_MEDIUM_CONFIDENCE", config.CLASSIFICATION_MEDIUM_CONFIDENCE),
        ("CLASSIFICATION_LOW_CONFIDENCE", config.CLASSIFICATION_LOW_CONFIDENCE),
        ("MIN_RESOLUTION_QUALITY", config.MIN_RESOLUTION_QUALITY),
        ("MIN_QUALITY_SCORE_FOR_MEMORY", config.MIN_QUALITY_SCORE_FOR_MEMORY),
        ("WEB_SEARCH_MIN_CONFIDENCE", config.WEB_SEARCH_MIN_CONFIDENCE),
        ("SELF_SERVICE_MIN_CONFIDENCE", config.SELF_SERVICE_MIN_CONFIDENCE),
        ("SELF_SERVICE_VIABILITY_MIN", config.SELF_SERVICE_VIABILITY_MIN),
        ("MAX_ACCEPTABLE_RISK", config.MAX_ACCEPTABLE_RISK),
        ("IT_EXECUTION_MIN_CONFIDENCE", config.IT_EXECUTION_MIN_CONFIDENCE),
    ]

    for name, value in thresholds:
        if not isinstance(value, (int, float)):
            errors.append(f"Invalid {name}: {value} (must be numeric)")
        elif not 0 <= value <= 1:
            errors.append(f"Invalid {name}: {value} (must be between 0.0 and 1.0)")

    # ========== Validate Positive Integers ==========
    positive_ints: List[Tuple[str, int]] = [
        ("MAX_AUGMENTATION_ITERATIONS", config.MAX_AUGMENTATION_ITERATIONS),
        ("ARTICLE_RERANK_TOP_K", config.ARTICLE_RERANK_TOP_K),
        ("UIPATH_JOB_TIMEOUT", config.UIPATH_JOB_TIMEOUT),
        ("MAX_TOKENS", config.MAX_TOKENS),
        ("KNOWLEDGE_BASE_SEARCH_LIMIT", config.KNOWLEDGE_BASE_SEARCH_LIMIT),
        ("CONFLUENCE_MEMORY_SEARCH_LIMIT", config.CONFLUENCE_MEMORY_SEARCH_LIMIT),
        ("WEB_SEARCH_MAX_RESULTS", config.WEB_SEARCH_MAX_RESULTS),
        ("WEB_SEARCH_TOP_RESULTS_TO_USE", config.WEB_SEARCH_TOP_RESULTS_TO_USE),
        ("IT_ACTIONS_CACHE_TTL_MINUTES", config.IT_ACTIONS_CACHE_TTL_MINUTES),
    ]

    for name, value in positive_ints:
        if not isinstance(value, int):
            errors.append(f"Invalid {name}: {value} (must be integer)")
        elif value <= 0:
            errors.append(f"Invalid {name}: {value} (must be positive)")

    # ========== Validate Temperature Range ==========
    if not isinstance(config.LLM_TEMPERATURE, (int, float)):
        errors.append(f"Invalid LLM_TEMPERATURE: {config.LLM_TEMPERATURE} (must be numeric)")
    elif not 0 <= config.LLM_TEMPERATURE <= 1:
        errors.append(f"Invalid LLM_TEMPERATURE: {config.LLM_TEMPERATURE} (must be between 0.0 and 1.0)")

    # ========== Validate Model ID ==========
    if not config.BEDROCK_MODEL_ID or not isinstance(config.BEDROCK_MODEL_ID, str):
        errors.append(f"Invalid BEDROCK_MODEL_ID: {config.BEDROCK_MODEL_ID} (must be non-empty string)")

    # ========== Validate Score Weights Sum to 1.0 ==========
    if hasattr(config, 'SCORE_WEIGHTS'):
        weights_sum = sum(config.SCORE_WEIGHTS.values())
        if abs(weights_sum - 1.0) > 0.01:  # Allow small floating point errors
            errors.append(
                f"Invalid SCORE_WEIGHTS: sum is {weights_sum:.4f}, must be 1.0 "
                f"(weights: {config.SCORE_WEIGHTS})"
            )

    # ========== Validate Environment Value ==========
    valid_environments = ["dev", "test", "prod"]
    if config.ENVIRONMENT not in valid_environments:
        errors.append(
            f"Invalid ENVIRONMENT: '{config.ENVIRONMENT}' "
            f"(must be one of: {', '.join(valid_environments)})"
        )

    # ========== Report Results ==========
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join([f"  - {e}" for e in errors])
        raise ConfigurationError(error_msg)

    # Validation passed - log success
    logger.info("[OK] Configuration validation passed")
    logger.info(f"   Environment: {config.ENVIRONMENT}")
    logger.info(f"   Model: {config.BEDROCK_MODEL_ID}")
    logger.info(f"   Knowledge threshold: {config.KNOWLEDGE_SUFFICIENCY_THRESHOLD}")
    logger.info(f"   IT action threshold: {config.IT_ACTION_MATCH_THRESHOLD}")
    logger.info(f"   Max augmentation iterations: {config.MAX_AUGMENTATION_ITERATIONS}")


def validate_environment_variables() -> List[str]:
    """
    Check which required environment variables are missing.

    Returns:
        List of missing environment variable names
    """
    required = [
        "UIPATH_URL",
        "UIPATH_ACCESS_TOKEN",
        "UIPATH_ORGANIZATION_ID",
        "UIPATH_TENANT_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY"
    ]

    missing = [var for var in required if not os.getenv(var)]
    return missing


if __name__ == "__main__":
    # Allow running validation standalone
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    print("\n" + "=" * 80)
    print("CONFIGURATION VALIDATION")
    print("=" * 80 + "\n")

    try:
        # Check environment variables first
        missing_vars = validate_environment_variables()
        if missing_vars:
            print("[WARNING] Missing environment variables:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\nNote: Some validations may fail without these variables.\n")

        # Run full validation
        validate_config()

        print("\n" + "=" * 80)
        print("[SUCCESS] All configuration checks passed!")
        print("=" * 80 + "\n")

        sys.exit(0)

    except ConfigurationError as e:
        print("\n" + "=" * 80)
        print("[FAILED] Configuration validation failed:")
        print("=" * 80)
        print(f"\n{e}\n")

        print("=" * 80)
        print("Please fix the errors above and try again.")
        print("=" * 80 + "\n")

        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error during validation: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
