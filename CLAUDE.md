# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an **autonomous IT support ticket resolution agent** built using **UiPath LangGraph SDK** and **AWS Bedrock Claude models**. The agent integrates with UiPath Orchestrator for a complete enterprise IT support automation solution.

### Core Capabilities
1. **Multi-source knowledge synthesis**: Confluence Memory, FreshDesk Articles (with semantic filtering), Context Grounding, Web Search
2. **Autonomous re-routing**: Automatically triggers web search when internal knowledge is insufficient, loops back to KB when web finds specific topics
3. **Iterative refinement**: Gap analysis with targeted augmentation (max 2 iterations)
4. **Intelligent response classification**: Three-way decision tree (self-service, IT execution, investigation)
5. **IT Action matching**: Predefined IT actions with information extraction and validation
6. **Cost tracking**: Full token usage and cost tracking for all LLM operations across workflow

**Key integration pattern**: LangGraph agent → UiPath Python SDK → Orchestrator (Processes/Storage/Context Grounding) → External systems

## Development Commands

### Environment Setup
```bash
# Install Python 3.12 and create virtualenv using uv
uv python install 3.12
uv venv

# Install dependencies
uv pip install -e .
```

### UiPath CLI Commands
```bash
# Authenticate (writes credentials to .env)
uv run uipath auth

# Initialize/update uipath.json with schemas
uv run uipath init

# Interactive development mode (recommended)
uv run uipath dev

# Run the agent directly
uv run uipath run agent '{"ticket_id": "12345"}'

# Package and deploy
uv run uipath pack
uv run uipath publish
uv run uipath deploy
```

### Testing Individual Integrations
```bash
# Test UiPath process invocation (ticket retrieval)
python tests/test_uipath_get_job_data.py --ticket_id 12345

# Test Storage Bucket (IT actions download)
python tests/test_uipath_storage_bucket.py

# Test Context Grounding (knowledge base search)
python tests/test_uipath_context_grounding.py --query "How do I reset my password?"

# Test article fetching
python tests/test_uipath_fetch_articles.py --keywords "password reset"

# Test package output node
python tests/test_package_output.py
```

## Architecture

### Project Structure
```
src/
├── graph/                 # LangGraph workflow definition
│   ├── state.py          # GraphState and GraphOutput schemas
│   └── workflow.py       # Graph structure with nodes and edges
├── nodes/                 # LangGraph node implementations
│   ├── get_ticket_info.py              # Step 1: Retrieve ticket
│   ├── search_memory.py                # Step 2: Search Confluence
│   ├── check_it_actions.py             # Step 3: Match IT actions
│   ├── extract_it_action_data.py       # Step 4: Extract info
│   ├── knowledge_search.py             # Step 5: Multi-source search
│   ├── evaluate_knowledge_sufficiency.py # Step 5b: Evaluate scores
│   ├── web_search_node.py              # Step 5c: Web search
│   ├── extract_web_search_topics.py    # Step 5d: Topic extraction
│   ├── check_missing_information.py    # Step 6a: Gap analysis
│   ├── augment_knowledge.py            # Step 6b: Augmentation
│   ├── generate_ticket_response.py     # Step 7: Generate response
│   └── package_output.py               # Step 8: Package final output
├── integrations/          # UiPath SDK integrations
│   ├── uipath_get_job_data.py          # Process invocation & job polling
│   ├── uipath_storage_bucket.py        # Storage Bucket file downloads
│   ├── uipath_context_grounding.py     # Knowledge base RAG search
│   ├── uipath_fetch_articles.py        # FreshDesk article fetching
│   └── confluence_memory.py            # Confluence memory storage/search
├── utils/                 # Helper utilities
│   ├── llm_service.py                  # AWS Bedrock LLM service with cost tracking
│   ├── keyword_extraction.py           # LLM-based keyword extraction
│   ├── article_search.py               # Unified article search
│   ├── article_cache.py                # Article caching
│   ├── prompt_utils.py                 # Prompt building utilities
│   └── exceptions.py                   # Custom exception classes
├── config/
│   ├── config.py                       # Centralized configuration
│   ├── prompts.py                      # LLM prompt templates
│   ├── assets_loader.py                # UiPath Assets loader (example POC)
│   └── validation.py                   # Configuration validation module
├── workflows/             # Workflow orchestration components
│   ├── web_search_resolution.py        # Web search workflow (used by web_search_node)
│   └── ticket_article_workflow.py      # Legacy ticket+article workflow (UNUSED)
└── agent/                 # Empty directory (placeholder)

main.py                    # Entry point - exports graph
tests/                     # Integration tests with CLI tools
```

### Graph Workflow Architecture

The agent uses a **LangGraph StateGraph** with conditional routing for autonomous decision-making. The workflow has **12 nodes** with **5 conditional edges** for dynamic routing.

#### Workflow Flow Diagram

```
START
  ↓
[1] get_ticket_info
  ↓
[2] check_it_actions
  ├─[Match]──→ [3] extract_it_action_data ──→ [7] generate_ticket_response
  └─[No Match]→ [4] search_memory
                   ↓
                [5] knowledge_search
                   ↓
                [6] evaluate_knowledge_sufficiency
                   ├─[Sufficient]───→ [7] generate_ticket_response
                   └─[Insufficient]→ [8] web_search_node
                                        ↓
                                     [9] extract_web_search_topics
                                        ├─[Has Topics]───→ [10] augment_knowledge
                                        └─[No Topics]────→ [11] check_missing_information
                                                              ├─[Needs More]→ [10] augment_knowledge
                                                              └─[Complete]──→ [7] generate_ticket_response
                                                                 (loop back max 2x)
[7] generate_ticket_response
  ↓
[12] package_output
  ↓
END
```

#### Node Descriptions

| Node | Purpose | Key Outputs |
|------|---------|-------------|
| `get_ticket_info` | Retrieves ticket details from UiPath process | `ticket_info` (TicketInfo model) |
| `search_memory` | Searches Confluence for similar resolved tickets | `memory_results` (List[Dict]) |
| `check_it_actions` | Matches ticket to predefined IT actions | `is_it_action_match`, `it_action_match` |
| `extract_it_action_data` | Extracts required fields for IT action | `ticket_information`, `missing_information`, `clarifying_questions` |
| `knowledge_search` | Searches FreshDesk + Context Grounding, performs semantic re-ranking | `freshservice_articles`, `context_grounding_results`, `knowledge_sufficiency` |
| `evaluate_knowledge_sufficiency` | Evaluates best score against threshold (0.7) | `knowledge_sufficiency` (is_sufficient, best_score, best_source) |
| `web_search_node` | Performs web search fallback | `web_search_resolution` (resolution_steps, sources, confidence) |
| `extract_web_search_topics` | LLM extracts specific topics from web results | `web_search_topics` (has_topics, topics, targeted_kb_queries) |
| `check_missing_information` | LLM gap analysis of collected knowledge | `missing_information_check` (needs_augmentation, missing_topics, targeted_queries) |
| `augment_knowledge` | Re-queries KB with targeted queries, tracks iterations | Updated `context_grounding_results`, `augmentation_iteration`, `augmentation_source` |
| `generate_ticket_response` | Generates response based on classification | One of: `ticket_response`, `it_solution_steps`, `it_further_investigation_actions` |
| `package_output` | Packages final output matching GraphOutput schema | `final_output` (structured dict) |

#### Conditional Routing Logic

1. **After IT Action Check** (`route_after_it_action_check`):
   - If `is_it_action_match=True` → extract_it_action_data
   - If `is_it_action_match=False` → search_memory

2. **After Knowledge Evaluation** (`route_after_knowledge_evaluation`):
   - If `knowledge_sufficiency.is_sufficient=True` (best score > 0.7) → generate_ticket_response
   - If `knowledge_sufficiency.is_sufficient=False` → web_search_node

3. **After Web Search Topics** (`route_after_web_search_topics`):
   - If `web_search_topics.has_topics=True` → augment_knowledge
   - If `web_search_topics.has_topics=False` → check_missing_information

4. **After Missing Info Check** (`route_after_missing_info_check`):
   - If `missing_information_check.needs_augmentation=True` AND `augmentation_iteration < 2` → augment_knowledge
   - Otherwise → generate_ticket_response

5. **After Augmentation** (`route_after_augmentation`):
   - Always → check_missing_information (creates loop for iterative refinement)

### State Management

The `GraphState` (TypedDict) tracks the complete lifecycle:

```python
class GraphState(TypedDict):
    # Input
    ticket_id: str

    # Step 1: Ticket info
    ticket_info: TicketInfo  # Dynamically created from config

    # Step 2: Memory search
    memory_results: NotRequired[List[Dict]]

    # Step 3: IT action matching
    is_it_action_match: bool
    it_action_match: NotRequired[Dict]

    # Step 4: IT action data extraction
    ticket_information: NotRequired[Dict]
    missing_information: NotRequired[List[str]]
    clarifying_questions: NotRequired[List[str]]
    can_proceed_with_it_action: NotRequired[bool]

    # Step 5: Knowledge search
    freshservice_articles: NotRequired[List[Dict]]
    context_grounding_results: NotRequired[List[Dict]]
    knowledge_sufficiency: NotRequired[Dict]

    # Step 5b: Web search
    web_search_resolution: NotRequired[Dict]

    # Step 5c: Topic extraction
    web_search_topics: NotRequired[Dict]

    # Step 6: Gap analysis & augmentation
    missing_information_check: NotRequired[Dict]
    augmentation_iteration: NotRequired[int]
    augmentation_source: NotRequired[str]

    # Step 7: Final response (one of three)
    ticket_response: NotRequired[str]  # Client-facing
    it_solution_steps: NotRequired[str]  # IT execution
    it_further_investigation_actions: NotRequired[str]  # Investigation

    # Step 8: Packaged output
    final_output: NotRequired[Dict]

    # Cost tracking
    llm_costs: Dict[str, Any]

    # Metadata
    metadata: NotRequired[Dict]
```

### UiPath Integrations (5 Total)

#### 1. Process Invocation (`uipath_get_job_data.py`)
**Purpose**: Invoke UiPath processes to retrieve ticket data from external systems (FreshDesk)

**Pattern**:
```python
from src.integrations.uipath_get_job_data import fetch_ticket_from_uipath

# In a LangGraph node
async def get_ticket_info(state):
    ticket_info = await fetch_ticket_from_uipath(state['ticket_id'])
    # Returns TicketInfo with: ticket_id, description, category, subject, requester, requester_email
    return {"ticket_info": ticket_info}
```

**Key Implementation Details**:
- **Dynamic model generation**: `TicketInfo` Pydantic model is created dynamically from `config.UIPATH_OUTPUT_MAPPING`
- **Job polling**: Waits for job completion with configurable timeout (default: 30s)
- **Output parsing**: Handles both JSON string and dict output arguments
- **Configuration-driven**: Add/remove fields by updating config only

**Job Flow**:
1. `sdk.processes.invoke()` - Start process with input arguments
2. Poll `sdk.jobs.retrieve(job.key)` until state is `Successful/Faulted/Stopped`
3. Parse `job.output_arguments` using mapping from config
4. Return populated `TicketInfo` model

**Configuration** (in `config.py`):
```python
UIPATH_PROCESS_NAME = "getFreshTicket2.0"
UIPATH_INPUT_TICKET_ID = "in_TicketID"
UIPATH_OUTPUT_MAPPING = {
    "description": "out_TicketDescription",
    "category": "out_category",
    "subject": "out_subject",
    "requester": "out_ticketRequester",
    "requester_email": "out_requesterEmail"
}
```

#### 2. Storage Bucket (`uipath_storage_bucket.py`)
**Purpose**: Download IT action categories JSON from UiPath Storage Buckets

**Pattern**:
```python
from src.integrations.uipath_storage_bucket import fetch_it_actions_from_bucket

# In a LangGraph node
def check_it_actions(state):
    it_actions = fetch_it_actions_from_bucket()
    # Returns dict with 'it_human_actions' containing categories
    # Use for LLM classification against ticket
    return {"available_actions": it_actions['it_human_actions']}
```

**Key Implementation Details**:
- Downloads JSON file from bucket to temp file
- Parses JSON and cleans up temp file automatically
- Returns structured dict with IT action categories
- Used for agent classification and routing decisions

**IT Actions JSON Structure**:
```json
{
  "it_human_actions": [
    {
      "category_name": "Access Management",
      "actions": [
        {
          "action_name": "Grant Application Access",
          "description": "...",
          "ticket_information_required": ["user_email", "application_name"]
        }
      ]
    }
  ]
}
```

#### 3. Context Grounding (`uipath_context_grounding.py`)
**Purpose**: Search UiPath Context Grounding knowledge base for relevant IT articles

**Pattern**:
```python
from src.integrations.uipath_context_grounding import ContextGroundingIntegration

# In a LangGraph node
def knowledge_search(state):
    cg = ContextGroundingIntegration()
    results = cg.search(query=state['query'], number_of_results=5)
    # Returns list of ContextGroundingResult with: content, source, score, metadata
    return {"context_grounding_results": results}
```

**Key Implementation Details**:
- Searches semantic index with natural language queries
- Returns ranked results with relevance scores (0.0-1.0)
- Extracts content, source, and metadata from results
- Index name and folder path configured in `config.py`

**Configuration**:
```python
# Main KB index
index_name = "IT"
folder_path = "IT"

# Memory index (Confluence)
CONFLUENCE_MEMORY_INDEX_NAME = "IT Approved Tickets"
CONFLUENCE_MEMORY_FOLDER_PATH = "IT/IT01_support_agent_fresh"
```

#### 4. Article Fetching (`uipath_fetch_articles.py`)
**Purpose**: Invoke UiPath process to fetch FreshDesk articles by keywords

**Pattern**:
```python
from src.integrations.uipath_fetch_articles import fetch_articles_from_uipath

# In a LangGraph node
async def knowledge_search(state):
    articles = await fetch_articles_from_uipath(
        keywords="password reset",
        status="published"
    )
    # Returns List[ArticleInfo] with: id, title, description, created_at, updated_at
    return {"freshservice_articles": articles}
```

**Key Implementation Details**:
- Asynchronous job invocation and polling
- Parses delimited article strings from UiPath output
- Format: `"id: 123 --- title: Title --- description: Desc --- ..."`
- Configurable timeout and status filtering

**Configuration**:
```python
UIPATH_ARTICLE_PROCESS_NAME = "getFreshArticles"
UIPATH_INPUT_KEYWORDS = "in_keywords"
UIPATH_INPUT_STATUS = "in_status"
UIPATH_OUTPUT_ARTICLES = "out_ArticlesList"
```

#### 5. Confluence Memory (`confluence_memory.py`)
**Purpose**: Store and retrieve successful ticket resolutions in Confluence via Context Grounding

**Pattern**:
```python
from src.integrations.confluence_memory import search_past_resolutions

# In a LangGraph node
def search_memory(state):
    memory_results = search_past_resolutions(
        query=state['ticket_info'].description,
        number_of_results=3
    )
    # Returns List[ConfluenceMemoryResult] with similar past resolutions
    return {"memory_results": memory_results}
```

**Key Implementation Details**:
- **Storage**: Creates Confluence pages with structured metadata
- **Retrieval**: Searches Context Grounding index for similar past tickets
- **Page format**: Markdown with ticket info, response, metadata
- **Scoring**: Uses Context Grounding relevance scores

**TicketResolution Model**:
```python
class TicketResolution(BaseModel):
    ticket_id: str
    category: str
    subject: str
    requester: str
    date: str
    priority: str
    description: str
    response: str
    related_articles: List[Dict[str, str]]
    status: str = "successful"
```

**Configuration**:
```python
CONFLUENCE_MEMORY_INDEX_NAME = "IT Approved Tickets"
CONFLUENCE_MEMORY_FOLDER_PATH = "IT/IT01_support_agent_fresh"
CONFLUENCE_MEMORY_SEARCH_LIMIT = 3
CONFLUENCE_MEMORY_MIN_SCORE = 0.5
```

### LLM Service Architecture

The agent uses a centralized `LLMService` class for all Claude interactions:

**Key Features**:
- **AWS Bedrock via LangChain**: Uses `ChatBedrockConverse` for full tracing support
- **Token counting**: Tracks input/output tokens per operation
- **Cost calculation**: Calculates cost based on model pricing ($3/1M input, $15/1M output)
- **Multi-method tracking**: Callback → usage_metadata → response_metadata → estimation
- **Aggregated reporting**: Provides total costs and per-operation breakdown

**Pattern**:
```python
from src.utils.llm_service import get_llm_service

llm = get_llm_service()
response = await llm.invoke(
    prompt="Analyze this ticket...",
    max_tokens=800,
    temperature=0.3,
    system_message="You are an IT support specialist",
    operation_name="check_it_actions"  # For cost tracking
)

# Get cost summary
costs = llm.get_total_costs()
# Returns: total_tokens, input_tokens, output_tokens, estimated_cost_usd, breakdown
```

**Token Counting Methods** (in order of preference):
1. **Callback**: `TokenCountingCallback` captures from LLM response
2. **usage_metadata**: LangChain standard field
3. **response_metadata**: Bedrock-specific field
4. **Estimation**: Falls back to ~4 chars per token

**Cost Tracking**:
```python
# Per operation
{
    "operation": "check_it_actions",
    "input_tokens": 1500,
    "output_tokens": 200,
    "total_tokens": 1700,
    "cost_usd": 0.0075,
    "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "tracking_method": "callback"
}

# Aggregated
{
    "total_tokens": 25000,
    "input_tokens": 20000,
    "output_tokens": 5000,
    "estimated_cost_usd": 0.1350,
    "llm_calls_count": 8,
    "average_cost_per_call": 0.0169,
    "breakdown": [...]  # List of per-operation records
}
```

### Advanced Features

#### 1. Semantic Article Re-ranking
**Purpose**: Filter FreshDesk articles by relevance before analysis

**Implementation** (`knowledge_search.py:_rerank_articles_by_relevance`):
- LLM scores each article (0.0-1.0) against ticket
- Batch processing (10 articles per LLM call) to prevent truncation
- Filtering logic:
  - If ≤5 articles: Keep articles with score ≥ 0.5
  - If >5 articles: Sort by score, keep top 5

**Configuration**:
```python
ARTICLE_RERANK_TOP_K = 5           # Max articles to keep
ARTICLE_RERANK_MIN_SCORE = 0.5     # Minimum relevance threshold
```

#### 2. Knowledge Sufficiency Evaluation
**Purpose**: Determine if internal knowledge is sufficient to skip web search

**Implementation** (`evaluate_knowledge_sufficiency.py`):
- Checks best score across memory, FreshDesk articles, Context Grounding
- Compares against threshold (0.7)
- Routes to either generate_ticket_response or web_search_node

**Threshold Rationale**:
- Score > 0.7: High confidence, internal knowledge sufficient
- Score ≤ 0.7: Low/medium confidence, trigger web search

#### 3. Iterative Knowledge Augmentation
**Purpose**: Fill knowledge gaps through targeted re-queries

**Implementation** (Steps 5d → 6a → 6b → loop):
1. **Extract topics** from web search results (if specific topics found)
2. **Gap analysis**: LLM identifies missing information
3. **Augmentation**: Re-query Context Grounding with targeted queries
4. **Loop back**: Re-evaluate sufficiency (max 2 iterations)

**Augmentation Sources**:
- `web_topics`: Topics extracted from web search
- `missing_info`: Gaps identified by LLM analysis

**Configuration**:
```python
MAX_AUGMENTATION_ITERATIONS = 2  # Hardcoded in check_missing_information.py
```

#### 4. Three-Way Response Classification
**Purpose**: Route to appropriate response type based on content analysis

**Decision Tree** (`generate_ticket_response.py:_determine_response_type`):
1. **user_executable**: User can perform steps (VPN, restart, clear cache)
   → Output: `ticket_response` (client-facing)
2. **clear_it_instructions**: Clear IT execution steps exist (admin actions)
   → Output: `it_solution_steps` (IT team internal)
3. **requires_investigation**: Unclear/incomplete information
   → Output: `it_further_investigation_actions` (IT team internal)

**Key Insight**: This is CONTENT-based, not score-based. LLM analyzes whether steps are user-executable vs require IT/admin access.

**Memory Integration**: Confluence memory results include `## Solution Type` labels (Self-Service, IT Action, Investigation) that guide classification.

#### 5. IT Action Matching with Information Extraction
**Purpose**: Match ticket to predefined IT actions and extract required fields

**Implementation** (`check_it_actions.py` + `extract_it_action_data.py`):

**Step 1: Classification**
- LLM matches ticket against IT action categories from Storage Bucket
- Returns: `is_match`, `matched_action_name`, `confidence`, `reasoning`, `ticket_information_required`

**Step 2: Information Extraction**
- LLM extracts required fields from ticket description
- Identifies missing information
- Generates clarifying questions for missing fields

**Decision Logic**:
- If `can_proceed_with_it_action=False` → Send clarifying questions to user
- If `can_proceed_with_it_action=True` → Generate IT execution steps

#### 6. Prompt Management System
**Purpose**: Centralized prompt templates and builders

**Key Components** (`prompts.py`):
- `CATO_AGENT_GUIDELINES`: Core identity and operating context
- Response type prompts (self-service, IT execution, investigation)
- Classification criteria and examples
- JSON response instructions

**PromptBuilder Utility** (`prompt_utils.py`):
```python
prompt = (PromptBuilder("TASK TITLE")
    .add_section("Section 1", content1)
    .add_section("Section 2", content2)
    .set_response_format("json")  # or "text"
    .build())
```

### Configuration System (`src/config/config.py`)

All settings centralized in single config file:

#### UiPath Process Configuration
```python
UIPATH_PROCESS_NAME = "getFreshTicket2.0"
UIPATH_ARTICLE_PROCESS_NAME = "getFreshArticles"
UIPATH_BUCKET_NAME = "IT_Actions_Bucket"
UIPATH_JOB_TIMEOUT = 30
```

#### LLM Configuration
```python
BEDROCK_MODEL_ID = "anthropic.claude-sonnet-4-20250514-v1:0"
MAX_TOKENS = 4000
LLM_TEMPERATURE = 0.3
```

#### Knowledge Base Configuration
```python
KNOWLEDGE_SUFFICIENCY_THRESHOLD = 0.6  # NOT USED - hardcoded to 0.7
ARTICLE_RERANK_TOP_K = 5
ARTICLE_RERANK_MIN_SCORE = 0.5
```

#### Confluence Memory Configuration
```python
CONFLUENCE_MEMORY_INDEX_NAME = "IT Approved Tickets"
CONFLUENCE_MEMORY_FOLDER_PATH = "IT/IT01_support_agent_fresh"
CONFLUENCE_MEMORY_MIN_SCORE = 0.5
MIN_QUALITY_SCORE_FOR_MEMORY = 0.75
```

#### Web Search Configuration
```python
WEB_SEARCH_ENABLED = True
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_TRUSTED_DOMAINS = [...]  # Microsoft, Stack Overflow, etc.
WEB_SEARCH_TOP_RESULTS_TO_USE = 3
```

### Error Handling and Exceptions

**Custom Exception Classes** (`utils/exceptions.py`):
- `RecoverableError`: Non-fatal errors, workflow continues
- `LLMError`: LLM-specific errors (parsing, invocation)
- `IntegrationError`: UiPath SDK/external integration errors
- `ConfigurationError`: Missing config or credentials
- `FatalError`: Critical errors that halt workflow

**Error Handling Pattern**:
```python
try:
    result = await some_operation()
except LLMError as e:
    logger.warning(f"LLM failed (recoverable): {e.message}")
    # Return safe default, continue workflow
except FatalError as e:
    logger.error(f"Critical error: {e.message}")
    raise  # Halt workflow
```

### Utility Modules

#### Keyword Extraction (`utils/keyword_extraction.py`)
- `extract_keywords_with_llm()`: LLM-based keyword extraction for article search
- `extract_keywords_for_knowledge_search()`: Optimized queries for Context Grounding

#### Article Search (`utils/article_search.py`)
- `search_articles_unified()`: Unified article search with caching
- Local cache with fallback to UiPath process invocation

#### Prompt Utilities (`utils/prompt_utils.py`)
- `PromptBuilder`: Fluent API for building structured prompts
- `build_ticket_context()`: Formats ticket info for prompts
- `parse_llm_json_response()`: Safe JSON parsing with error handling
- `log_llm_response()`: Standardized logging for LLM responses

### Output Packaging

**Final Output Schema** (`GraphOutput` in `state.py`):
```python
{
    "ticket_info": TicketInfo,
    "ticket_analysis": Dict,  # IT action match or knowledge sufficiency
    "decision": Dict,  # Routing decisions and confidence
    "response_evaluation": Dict,  # Quality metrics (not yet implemented)
    "client_response": Optional[str],  # If self-service
    "it_execution_instructions": Optional[Dict],  # If IT execution
    "it_investigation_instructions": Optional[Dict],  # If investigation
    "solution_sources": Dict,  # Memory, articles, CG, web
    "llm_costs": Dict,  # Token usage and costs
    "metadata": Dict  # Timestamps, versions
}
```

**Packaging Logic** (`package_output.py`):
- Extracts final response from state (ticket_response, it_solution_steps, or it_further_investigation_actions)
- Maps to appropriate GraphOutput field
- Aggregates all knowledge sources
- Includes complete cost breakdown
- Adds metadata (timestamps, augmentation info)

## Key Implementation Patterns

### 1. Dynamic Model Generation (TicketInfo)
The `TicketInfo` model is created dynamically from config using Pydantic's `create_model()`:

```python
def _create_ticket_info_model():
    field_definitions = {'ticket_id': (str, ...)}  # Always required

    # Add fields from UIPATH_OUTPUT_MAPPING
    for field_name in config.UIPATH_OUTPUT_MAPPING.keys():
        default_value = config.UIPATH_OUTPUT_DEFAULTS.get(field_name, None)
        if default_value is not None:
            field_definitions[field_name] = (str, default_value)
        else:
            field_definitions[field_name] = (str, ...)

    return create_model('TicketInfo', **field_definitions)

TicketInfo = _create_ticket_info_model()
```

**Why this matters**: To add/remove ticket fields, only update `UIPATH_OUTPUT_MAPPING` and `UIPATH_OUTPUT_DEFAULTS` in config. The model regenerates automatically.

### 2. Job Polling with UiPath SDK
```python
# IMPORTANT: Use job.key (not job.id)
job = sdk.processes.invoke(
    name=process_name,
    input_arguments={input_arg_name: ticket_id},
    folder_path=folder_path if folder_path else None
)

# Poll with job.key as positional argument
job_status = sdk.jobs.retrieve(
    job.key,  # Positional, not key=job.key
    folder_path=folder_path if folder_path else None
)

# Check state
if job_status.state == "Successful":
    output = job_status.output_arguments
    # May be JSON string or dict - handle both
    if isinstance(output, str):
        output = json.loads(output)
```

### 3. Async/Await Pattern
All LLM operations and UiPath process invocations use async/await:

```python
async def node_function(state: GraphState) -> dict:
    # Async LLM call
    llm = get_llm_service()
    response = await llm.invoke(prompt, max_tokens=800)

    # Async UiPath process invocation
    articles = await fetch_articles_from_uipath(keywords)

    return {"results": response}
```

### 4. Configuration-Driven Development
When UiPath processes change:
1. Update `src/config/config.py` mappings
2. Run `uv run uipath init` to regenerate schemas
3. Test integration with test file
4. No code changes needed in integration layer

### 5. Centralized LLM Service
All nodes use shared LLM service instance for cost aggregation:

```python
from src.utils.llm_service import get_llm_service

# In any node
llm = get_llm_service()  # Gets global instance
response = await llm.invoke(prompt, operation_name="node_name")

# At end of workflow
costs = llm.get_total_costs()  # Aggregated across all operations
```

### 6. Safe JSON Parsing
All LLM JSON responses use safe parsing:

```python
from src.utils.prompt_utils import parse_llm_json_response

try:
    result = parse_llm_json_response(response, logger, "operation_name")
    # Handles: markdown fences, extra text, malformed JSON
except (json.JSONDecodeError, ValueError) as e:
    logger.error(f"Failed to parse: {e}")
    # Return safe default
    return default_value
```

## Environment Variables

Required in `.env` (created by `uv run uipath auth`):
- **UiPath**: `UIPATH_URL`, `UIPATH_ACCESS_TOKEN`, `UIPATH_ORGANIZATION_ID`, `UIPATH_TENANT_ID`
- **Optional**: `UIPATH_FOLDER_PATH` (defaults to "" if not set)
- **AWS Bedrock**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- **Model**: `BEDROCK_MODEL_ID` (defaults to Claude Sonnet 4)

UiPath SDK reads these automatically on initialization.

## Important Technical Details

### UiPath SDK Initialization
```python
from uipath import UiPath
from dotenv import load_dotenv

load_dotenv()  # Load .env file
sdk = UiPath()  # Reads from environment automatically
```

- SDK reads credentials from environment variables automatically
- Initialize with `UiPath()` - no parameters needed
- Always call `load_dotenv()` first in standalone scripts

### Job Key vs Job ID
**Critical**: UiPath SDK uses `job.key` (string) not `job.id` (number)
- After invoking: `job = sdk.processes.invoke(...)`
- Use `job.key` for all subsequent operations
- Pass as positional arg: `sdk.jobs.retrieve(job.key, folder_path=...)`

### Output Argument Handling
UiPath processes can return output arguments as:
1. JSON string: `'{"field": "value"}'`
2. Dictionary: `{"field": "value"}`

Always handle both:
```python
if isinstance(output_args, str):
    output_args = json.loads(output_args)
```

### Folder Path Handling
Folder path can be:
- Empty string `""` - Uses default folder
- Folder name like `"IT"` or `"Shared"`
- `None` - Don't pass to SDK

Pattern: `folder_path=self.config.folder_path if self.config.folder_path else None`

### LangGraph Execution
- Graph must be compiled before running: `graph = builder.compile()`
- Entry point exposed in `langgraph.json` under `graphs` key
- Input/output schemas in `uipath.json` auto-generated by `uipath init`

## Development Workflow

1. **Modify graph logic** in `src/graph/workflow.py` or add nodes in `src/nodes/`
2. **Update configuration** in `src/config/config.py` if needed
3. **Test integrations** using test files in `tests/` directory
4. **Run `uv run uipath init`** to update schemas in `uipath.json`
5. **Test locally** with `uv run uipath dev` (interactive) or `uv run uipath run agent '{"ticket_id": "12345"}'`
6. **Deploy** via `uv run uipath deploy` when ready

## Troubleshooting

### UiPath Authentication Issues
- Run `uv run uipath auth` to refresh credentials
- Verify `.env` has `UIPATH_ACCESS_TOKEN`, `UIPATH_URL`, etc.
- Check token hasn't expired

### Job Invocation Failures (404 Not Found)
- Check `UIPATH_PROCESS_NAME` matches exact Release name in Orchestrator
- Verify `UIPATH_FOLDER_PATH` matches target folder
- Ensure process is published to the correct folder

### Job Invocation Failures (Unexpected keyword argument)
- Use `job.key` not `job.id`
- Pass job key as positional argument: `sdk.jobs.retrieve(job_key, folder_path=...)`
- Don't use `key=job_key` as keyword argument

### Output Parsing Errors
- Compare actual UiPath output args with `UIPATH_OUTPUT_MAPPING`
- Check logs for "Output argument not found" warnings
- Handle both JSON string and dict: `if isinstance(output_args, str): output_args = json.loads(output_args)`
- Update mapping or defaults in `src/config/config.py`

### Storage Bucket Download Fails
- Verify `UIPATH_BUCKET_NAME` and `IT_ACTIONS_JSON_PATH` in config
- Check file exists in bucket with correct path
- Ensure folder path is correct (empty string for default folder)

### Context Grounding Search Fails
- Verify index name exists in folder
- Check index has been populated with documents
- Use test file to verify: `python tests/test_uipath_context_grounding.py --query "test"`

### LangGraph Schema Mismatches
- Run `uv run uipath init` after changing Pydantic models
- Ensure `GraphState` and `GraphOutput` models match graph's input/output types
- Check `uipath.json` has correct schemas

### LLM Cost Tracking Not Working
- Check AWS credentials in `.env`
- Verify `ChatBedrockConverse` is being used (not `UiPathChat` direct)
- Review logs for "[Cost Tracking]" entries
- If estimation used, check for warning logs

### Article Re-ranking Failures
- Check batch size (default 10) - reduce if LLM truncates responses
- Review semantic scores in logs
- Adjust `ARTICLE_RERANK_MIN_SCORE` threshold if too many/few articles

### Packaging Failures (path is on mount '\\\\.\\nul')
**Error**: `Failed to create package: path is on mount '\\\\.\\nul', start on mount 'C:'`

**Cause**: A file named `nul` exists in the project directory. On Windows, `nul` is a reserved device name (like `/dev/null` on Linux), causing the UiPath packaging process to fail with a mount error.

**Solution**:
1. Check for the file: `ls -la | grep -i nul` or look in project root
2. Delete the file: `rm nul`
3. Remove `nul` from `.gitignore` if present (it shouldn't be tracked)
4. Retry: `uv run uipath pack`

**Prevention**: Avoid redirecting command output to `nul` in Windows. Use `> $null` (PowerShell) or `> /dev/null` (Git Bash) instead.

## Testing Strategy

### Integration Tests
Each UiPath integration has a dedicated test file:
- `tests/test_uipath_get_job_data.py` - Process invocation
- `tests/test_uipath_storage_bucket.py` - Storage Bucket downloads
- `tests/test_uipath_context_grounding.py` - Context Grounding search
- `tests/test_uipath_fetch_articles.py` - Article fetching
- `tests/test_package_output.py` - Output packaging

Run these tests to verify connectivity and configuration before building agent logic.

### Test Workflow
1. Authenticate: `uv run uipath auth`
2. Run integration tests individually to verify each component
3. Build/modify LangGraph agent in `src/graph/workflow.py`
4. Test full agent with `uv run uipath dev`
5. Monitor logs for cost tracking and routing decisions

## Performance Considerations

### LLM Calls
- Agent makes 6-12 LLM calls per ticket (depending on routing)
- Average cost: $0.10-0.20 per ticket (Claude Sonnet 4)
- Token usage: 20k-40k tokens per ticket

### Optimization Opportunities
1. **Cache article search results** - Already implemented with `article_cache.py`
2. **Reduce augmentation iterations** - Currently max 2, consider 1 for cost savings
3. **Batch article re-ranking** - Already implemented (10 per batch)
4. **Skip web search for high-confidence tickets** - Already implemented (threshold 0.7)

### Scalability
- Graph is stateless - can handle concurrent executions
- UiPath SDK handles connection pooling
- Consider caching IT actions JSON (currently fetched per ticket)

## Reference Documentation

- **UiPath Python SDK**: https://github.com/UiPath/uipath-python
- **UiPath SDK docs**: https://uipath.github.io/uipath-python/
- **LangGraph docs**: https://langchain-ai.github.io/langgraph/
- **AWS Bedrock pricing**: https://aws.amazon.com/bedrock/pricing/
- **Claude models**: https://www.anthropic.com/api

## Recent Changes and Updates

### Latest Features
- **Confluence memory integration** for storing/retrieving past resolutions
- **Semantic article re-ranking** with LLM-based relevance scoring
- **Three-way response classification** (user-executable, IT execution, investigation)
- **Iterative knowledge augmentation** with gap analysis
- **Comprehensive cost tracking** across all LLM operations
- **Structured investigation responses** using Pydantic models

### Known Limitations
1. **Response evaluation not implemented** - `response_evaluation` field in GraphOutput is placeholder
2. **Confluence page creation not tested** - `create_confluence_page()` method exists but may need SDK updates
3. **Knowledge sufficiency threshold hardcoded** - Config has 0.6, code uses 0.7
4. **IT action self-service detection** - Not yet implemented in `_generate_it_action_response`
5. **Max augmentation iterations hardcoded** - Not configurable via config.py

### Future Enhancements
- Dynamic response evaluation with quality scoring
- Confluence page creation for successful resolutions
- Configurable augmentation limits
- IT action self-service vs execution classification
- Enhanced caching strategy for IT actions
- Multi-tenant support with folder-based isolation
