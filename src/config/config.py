"""
Configuration settings for the IT Support Agent
"""

# ========== UIPATH CONFIGURATION ==========

# IMPORTANT: Update this with your actual UiPath process name from Orchestrator
UIPATH_PROCESS_NAME = "getFreshTicket2.0"  # TODO: Replace with actual process name


# Example: "FreshDesk_GetTicketInfo" or "IT_Support_TicketRetrieval"
# You can find this in UiPath Orchestrator under Processes

# Job execution timeout (seconds)
UIPATH_JOB_TIMEOUT = 30

# ========== UIPATH INPUT/OUTPUT MAPPING ==========

# Input argument name for ticket ID
UIPATH_INPUT_TICKET_ID = "in_TicketID"

# Output argument names mapping
# Update these to match your UiPath process output arguments
UIPATH_OUTPUT_MAPPING = {
    "description": "out_TicketDescription",
    "category": "out_category",
    "subject": "out_subject",
    "requester": "out_ticketRequester",
    "requester_email": "out_requesterEmail"
    # Note: relevant_articles removed - not provided by UiPath process
}

# Optional: Default values if output argument is missing
UIPATH_OUTPUT_DEFAULTS = {
    "description": "No description provided",
    "category": "Uncategorized",
    "subject": "No subject",
    "requester": "Unknown",
    "requester_email": "unknown@example.com"
}

# ========== ARTICLE FETCHING CONFIGURATION ==========

# UiPath process for fetching articles from FreshDesk
UIPATH_ARTICLE_PROCESS_NAME = "getFreshArticles"  # TODO: Replace with actual process name

# Input argument names for article fetching
UIPATH_INPUT_KEYWORDS = "in_keywords"
UIPATH_INPUT_STATUS = "in_status"
UIPATH_DEFAULT_STATUS = "published"

# Output argument name for articles list
UIPATH_OUTPUT_ARTICLES = "out_ArticlesList"

# ========== STORAGE BUCKET CONFIGURATION ==========

# UiPath Storage Bucket name for IT actions
UIPATH_BUCKET_NAME = "IT_Actions_Bucket"

# Path to IT actions JSON file in the storage bucket
IT_ACTIONS_JSON_PATH = "it_actions.json"

# IT Actions cache TTL in minutes (to avoid fetching on every ticket)
IT_ACTIONS_CACHE_TTL_MINUTES = 60

# ========== CLASSIFICATION CONFIGURATION ==========

# Classification confidence thresholds
CLASSIFICATION_HIGH_CONFIDENCE = 0.8
CLASSIFICATION_MEDIUM_CONFIDENCE = 0.6
CLASSIFICATION_LOW_CONFIDENCE = 0.3

# IT Action match threshold (minimum confidence to classify as IT action)
IT_ACTION_MATCH_THRESHOLD = 0.90

# ========== RESOLUTION CONFIGURATION ==========

# Maximum number of resolution retry attempts
MAX_RESOLUTION_RETRIES = 3

# Minimum quality score for acceptable resolution
MIN_RESOLUTION_QUALITY = 0.7

# ========== LLM CONFIGURATION ==========

# UiPath Chat Model ID
# Available models in UiPath Normalized API:
# - anthropic.claude-sonnet-4-20250514-v1:0 (Claude Sonnet 4)
# - anthropic.claude-sonnet-4-5-20250929-v1:0 (Claude Sonnet 4.5)
# - anthropic.claude-3-5-sonnet-20241022-v2:0 (Claude 3.5 Sonnet)
# - anthropic.claude-3-7-sonnet-20250219-v1:0 (Claude 3.7 Sonnet)
BEDROCK_MODEL_ID = "anthropic.claude-sonnet-4-20250514-v1:0"

# Max tokens for LLM responses
MAX_TOKENS = 4000

# Temperature for LLM (0.0 - 1.0)
LLM_TEMPERATURE = 0.3

# ========== KNOWLEDGE BASE CONFIGURATION ==========

# UiPath Context Grounding settings (if using)
CONTEXT_GROUNDING_ENABLED = True
KNOWLEDGE_BASE_SEARCH_LIMIT = 5

# Knowledge sufficiency threshold
# If best score > this threshold, skip web search
# If best score ≤ this threshold, trigger web search
# NOTE: Updated from 0.6 to 0.7 to match code implementation
KNOWLEDGE_SUFFICIENCY_THRESHOLD = 0.7

# ========== CONFLUENCE MEMORY CONFIGURATION ==========

# Confluence Memory settings for storing successful resolutions
CONFLUENCE_MEMORY_ENABLED = True
CONFLUENCE_MEMORY_INDEX_NAME = "IT Approved Tickets"
CONFLUENCE_MEMORY_FOLDER_PATH = "IT/IT01_support_agent_fresh"
CONFLUENCE_MEMORY_SEARCH_LIMIT = 3

# Minimum score threshold for memory search results
# Results with score < this threshold will be filtered out
CONFLUENCE_MEMORY_MIN_SCORE = 0.5

# High confidence threshold for memory results (for logging purposes)
CONFLUENCE_MEMORY_HIGH_SCORE = 0.85

# Minimum quality score to store resolution in Confluence memory
MIN_QUALITY_SCORE_FOR_MEMORY = 0.75

# ========== WEB SEARCH CONFIGURATION ==========

# Enable web search fallback
WEB_SEARCH_ENABLED = True
WEB_SEARCH_MAX_RESULTS = 5

# Trusted IT support domains (whitelist approach for security)
WEB_SEARCH_TRUSTED_DOMAINS = [
    # Microsoft Official
    "microsoft.com",
    "docs.microsoft.com",
    "support.microsoft.com",
    "learn.microsoft.com",
    "techcommunity.microsoft.com",

    # Developer Resources
    "stackoverflow.com",
    "github.com",
    "superuser.com",
    "serverfault.com",

    # IT Knowledge Bases
    "atlassian.com",
    "servicenow.com",

    # Vendor Support Sites
    "cisco.com",
    "vmware.com",
    "redhat.com",
    "ubuntu.com",
    "dell.com",
    "hp.com",

    # General IT Resources
    "howtogeek.com",
    "techrepublic.com",
]

# Web search security and performance settings
WEB_SEARCH_MAX_CONTENT_LENGTH = 1000  # Max characters per result
WEB_SEARCH_TIMEOUT_SECONDS = 10       # Timeout for search API calls
WEB_SEARCH_RATE_LIMIT_DELAY = 1       # Seconds between requests

# Claude-enhanced search settings
WEB_SEARCH_TOP_RESULTS_TO_USE = 3     # After ranking, use top N results
WEB_SEARCH_MIN_CONFIDENCE = 0.6       # Minimum confidence score for resolution

# ========== RESPONSE EVALUATION CONFIGURATION ==========

# Context Grounding index and folder for evaluation criteria
RESPONSE_EVALUATION_INDEX_NAME = "Response Evaluation Criteria"
RESPONSE_EVALUATION_FOLDER_PATH = "IT/IT01_support_agent_fresh"

# Dynamic scoring thresholds (all 0-1 scale)
# These thresholds determine routing logic in Step 8b

# Self-service routing thresholds
SELF_SERVICE_MIN_CONFIDENCE = 0.5       # Minimum overall_confidence for self-service
SELF_SERVICE_VIABILITY_MIN = 0.4        # Minimum self_service_viability score
MAX_ACCEPTABLE_RISK = 0.30               # Maximum risk_score allowed for self-service

# IT execution routing thresholds
IT_EXECUTION_MIN_CONFIDENCE = 0.60       # Minimum overall_confidence for IT execution
# Uses MAX_ACCEPTABLE_RISK from above

# IT investigation routing
# Anything below IT_EXECUTION_MIN_CONFIDENCE or above MAX_ACCEPTABLE_RISK goes to investigation

# Score weighting for overall_confidence calculation
# Must sum to 1.0
SCORE_WEIGHTS = {
    "quality_score": 0.20,
    "completeness_score": 0.20,
    "self_service_viability": 0.20,
    "risk_score": 0.15,              # Inverted (1 - risk_score) in calculation
    "source_confidence": 0.15,
    "actionability_score": 0.10
}

# ========== ARTICLE SEMANTIC RE-RANKING CONFIGURATION ==========

# Semantic re-ranking for FreshDesk articles
ARTICLE_RERANK_TOP_K = 5           # Number of articles to keep if >5 articles
ARTICLE_RERANK_MIN_SCORE = 0.5     # Minimum relevance score threshold

# ========== AUGMENTATION CONFIGURATION ==========

# Maximum number of knowledge augmentation iterations
# Used in check_missing_information.py to prevent infinite loops
MAX_AUGMENTATION_ITERATIONS = 2

# Minimum quality score for augmented results
# Results below this threshold are filtered out
AUGMENTATION_QUALITY_THRESHOLD = 0.5

# ========== KNOWLEDGE SEARCH CONFIGURATION ==========

# Minimum confidence score to draft response from knowledge base
# Used in knowledge_search.py to validate LLM responses
KNOWLEDGE_SEARCH_MIN_CONFIDENCE = 0.70

# ========== LOGGING CONFIGURATION ==========

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Log file path
LOG_FILE_PATH = "logs/agent.log"

# ========== ENVIRONMENT PROFILES ==========

import os

# Get environment from environment variable (default: prod)
ENVIRONMENT = os.getenv("ENVIRONMENT", "test")

# Environment-specific overrides
# These allow tuning thresholds per environment without code changes
if ENVIRONMENT == "dev":
    # Dev: Lower thresholds for easier testing
    KNOWLEDGE_SUFFICIENCY_THRESHOLD = 0.5
    MAX_AUGMENTATION_ITERATIONS = 1
    IT_ACTION_MATCH_THRESHOLD = 0.80
    KNOWLEDGE_SEARCH_MIN_CONFIDENCE = 0.60
    LLM_TEMPERATURE = 0.5  # More creative responses for testing

elif ENVIRONMENT == "prod":
    # Test: Medium thresholds for validation
    KNOWLEDGE_SUFFICIENCY_THRESHOLD = 0.6
    MAX_AUGMENTATION_ITERATIONS = 2
    IT_ACTION_MATCH_THRESHOLD = 0.85
    KNOWLEDGE_SEARCH_MIN_CONFIDENCE = 0.65
    LLM_TEMPERATURE = 0.35

# prod uses the default values defined above
# No need for else block - values already set