"""
UiPath Assets Configuration Loader - EXAMPLE IMPLEMENTATION

This is a working example showing how to load configuration from UiPath Assets.
Currently only migrates KNOWLEDGE_SUFFICIENCY_THRESHOLD as proof-of-concept.

Usage:
    from src.config.assets_loader import get_knowledge_threshold

    threshold = get_knowledge_threshold()
    # Returns value from UiPath Asset "ITAgent_KnowledgeThreshold"
    # Falls back to config.py if asset not found
"""

import os
import logging
from typing import Optional
from src.config import config

logger = logging.getLogger(__name__)

# Cache for loaded values (avoid fetching asset on every call)
_threshold_cache: Optional[float] = None


def get_knowledge_threshold() -> float:
    """
    Get KNOWLEDGE_SUFFICIENCY_THRESHOLD from UiPath Assets.

    Tries to load from UiPath Asset "ITAgent_KnowledgeThreshold" (Text type).
    Falls back to config.py if:
    - Asset doesn't exist
    - UiPath SDK not available
    - Any error occurs

    Returns:
        float: Knowledge sufficiency threshold (0.0-1.0)

    Example:
        threshold = get_knowledge_threshold()
        if best_score > threshold:
            # Knowledge is sufficient
    """
    global _threshold_cache

    # Return cached value if available
    if _threshold_cache is not None:
        return _threshold_cache

    # Try to load from UiPath Assets
    try:
        from uipath import UiPath

        sdk = UiPath()
        logger.info("[Assets] Attempting to load ITAgent_KnowledgeThreshold from UiPath Assets")

        # Retrieve asset (no folder_path needed - uses current tenant)
        asset = sdk.assets.retrieve("ITAgent_KnowledgeThreshold")

        # Parse value from Text asset (stored as string)
        threshold = float(asset.value)

        # Validate range
        if not 0 <= threshold <= 1:
            logger.warning(
                f"[Assets] Invalid threshold value {threshold} in asset "
                f"(must be 0.0-1.0). Falling back to config.py"
            )
            threshold = config.KNOWLEDGE_SUFFICIENCY_THRESHOLD
        else:
            logger.info(
                f"[Assets] Successfully loaded threshold from asset: {threshold}"
            )

        # Cache the value
        _threshold_cache = threshold
        return threshold

    except ImportError:
        # UiPath SDK not available (local development)
        logger.debug("[Assets] UiPath SDK not available, using config.py")
        return config.KNOWLEDGE_SUFFICIENCY_THRESHOLD

    except Exception as e:
        # Asset not found or other error - fall back to config
        logger.info(
            f"[Assets] Could not load from UiPath Assets ({type(e).__name__}: {e}). "
            f"Falling back to config.py"
        )
        return config.KNOWLEDGE_SUFFICIENCY_THRESHOLD


def reload_threshold() -> float:
    """
    Force reload threshold from UiPath Assets (clears cache).

    Useful for long-running processes that need to pick up
    asset changes without restarting.

    Returns:
        float: Freshly loaded threshold value
    """
    global _threshold_cache
    _threshold_cache = None
    return get_knowledge_threshold()


def get_threshold_source() -> str:
    """
    Get the source of the current threshold value.

    Returns:
        str: "UiPath Assets" or "config.py"
    """
    try:
        from uipath import UiPath
        sdk = UiPath()
        asset = sdk.assets.retrieve("ITAgent_KnowledgeThreshold")
        return "UiPath Assets"
    except:
        return "config.py"


# ============================================================================
# HOW TO CREATE THE ASSET IN ORCHESTRATOR
# ============================================================================
"""
STEP-BY-STEP GUIDE:

1. Navigate to UiPath Orchestrator (your tenant - Dev/Test/Prod)
2. Go to: Tenant → Assets
3. Click "Add Asset"
4. Fill in:
   - Name: ITAgent_KnowledgeThreshold
   - Type: Text  ← IMPORTANT: Use Text type (not Integer)
   - Value: Enter threshold as text:
     * Dev Tenant: "0.5"
     * Test Tenant: "0.6"
     * Prod Tenant: "0.7"
   - Description: Knowledge sufficiency threshold for IT agent
5. Click "Create"

WHY TEXT TYPE?
- Orchestrator only has Integer and Text asset types
- Float values must be stored as Text (e.g., "0.7")
- The code converts it: float(asset.value)

PER-TENANT CONFIGURATION:
Each tenant gets its own asset with different value:

┌─────────────────────────────────────────────┐
│ Dev Tenant Assets                           │
├─────────────────────────────────────────────┤
│ ITAgent_KnowledgeThreshold = "0.5"  (Text) │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Test Tenant Assets                          │
├─────────────────────────────────────────────┤
│ ITAgent_KnowledgeThreshold = "0.6"  (Text) │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Prod Tenant Assets                          │
├─────────────────────────────────────────────┤
│ ITAgent_KnowledgeThreshold = "0.7"  (Text) │
└─────────────────────────────────────────────┘

The agent automatically uses the asset from whatever tenant it runs in!
No environment variables needed.

FALLBACK BEHAVIOR:
If asset doesn't exist → Uses value from config.py
Perfect for local development where UiPath Assets aren't available.
"""
