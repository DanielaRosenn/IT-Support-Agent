# IT Support Agent - Autonomous Ticket Resolution

An intelligent IT support agent built with **UiPath LangGraph SDK** that autonomously resolves IT tickets through multi-source knowledge integration, semantic article filtering, and self-service detection.

## 🚀 Key Features

### **Autonomous Knowledge Re-Routing**
- Automatically triggers external web search when internal knowledge is insufficient (score ≤ 0.8)
- Intelligently loops back to internal KB when web search identifies specific topics
- Iterative refinement with gap analysis (max 2 iterations)

### **Semantic Article Filtering **
- LLM-based relevance scoring for FreshDesk articles
- Adaptive logic:
  - **≤5 articles**: Keep only those with score ≥ 0.7
  - **>5 articles**: Sort by score, keep top 5
- Reduces noise from keyword-only search

### **Self-Service Detection **
- LLM evaluates if user can self-service the issue
- Generates appropriate response type:
  - **Client-facing instructions** (self-service)
  - **IT execution steps** (requires admin)
  - **Investigation actions** (incomplete info)

### **Multi-Source Integration**
Synthesizes information from:
- **Confluence Memory**: Past resolved tickets
- **Context Grounding**: Internal documentation
- **FreshDesk Articles**: Semantically filtered
- **Web Search**: External trusted sources
- **Augmented KB**: Targeted re-querying

---

## 📋 Prerequisites

- **Python 3.11+**
- **UiPath Orchestrator** access
- **uv** package manager (recommended)
- **UiPath Python SDK**

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd IT01_support_agent_fresh
```

### 2. Set Up Environment
```bash
# Install uv (if not installed)
pip install uv

# Install Python 3.12 and create virtual environment
uv python install 3.12
uv venv

# Install dependencies
uv pip install -e .
```

### 3. Configure UiPath Credentials
```bash
# Authenticate with UiPath Orchestrator (creates .env file)
uv run uipath auth
```

### 4. Configure Application Settings
Update `src/config/config.py` with your environment:
- `UIPATH_PROCESS_NAME` - Your UiPath process for fetching tickets
- `UIPATH_BUCKET_NAME` - Storage bucket for IT actions
- Confluence Memory settings
- Context Grounding settings
- Web search trusted domains

---

## 🚀 Quick Start

### Initialize Schemas
```bash
# Run in PowerShell (not Bash) to avoid encoding issues
uv run uipath init
```

### Run Agent Locally
```bash
# Interactive mode
uv run uipath dev

# Direct invocation with ticket ID
uv run uipath run agent '{"ticket_id": "42924"}'
```

### Deploy to UiPath Orchestrator
```bash
# Package agent
uv run uipath pack

# Publish to Orchestrator
uv run uipath publish

# Deploy
uv run uipath deploy
```

---

## 📐 Architecture

### Workflow Overview

```
Ticket Input → Get Ticket Info → Check IT Actions
                                      ↓
                        ┌─────────────┴──────────────┐
                        ↓ Match                      ↓ No Match
                   Extract Data              Search Memory
                        ↓                           ↓
                        ↓                    Knowledge Search
                        ↓                    (with semantic filtering)
                        ↓                           ↓
                        ↓                Evaluate Sufficiency
                        ↓                           ↓
                        ↓              ┌────────────┴───────────┐
                        ↓              ↓ Sufficient             ↓ Insufficient
                        ↓         (Skip to response)        Web Search
                        ↓                                       ↓
                        ↓                              Extract Topics
                        ↓                                       ↓
                        ↓                            ┌──────────┴──────────┐
                        ↓                            ↓ Has Topics          ↓ No Topics
                        ↓                     Augment Knowledge    Check Missing Info
                        ↓                            ↓                     ↓
                        ↓                            └──────→ Loop ←───────┘
                        ↓                                    (max 2 iterations)
                        ↓                                       ↓
                        └───────────────────────────────────────┘
                                            ↓
                                 Generate Response
                        (Self-Service | IT Execution | Investigation)
```

See [`docs/WORKFLOW_ARCHITECTURE.md`](docs/WORKFLOW_ARCHITECTURE.md) for detailed architecture documentation.

---

## 📂 Project Structure

```
src/
├── nodes/                           # LangGraph nodes
│   ├── get_ticket_info.py          # Fetch ticket from UiPath
│   ├── check_it_actions.py         # IT action classification
│   ├── search_memory.py            # Confluence memory search
│   ├── knowledge_search.py         # Article + CG search (with semantic filtering)
│   ├── evaluate_knowledge_sufficiency.py    # Knowledge score evaluation
│   ├── web_search_node.py          # External web search
│   ├── extract_web_search_topics.py    # Topic extraction from web
│   ├── check_missing_information.py    # Gap analysis
│   ├── augment_knowledge.py        # Targeted KB re-querying
│   └── generate_ticket_response.py # Response generation (with self-service detection)
│
├── graph/
│   ├── state.py                    # GraphState, GraphInput, GraphOutput schemas
│   └── workflow.py                 # Main workflow with routing logic
│
├── integrations/                   # UiPath SDK integrations
│   ├── uipath_get_job_data.py     # Process invocation
│   ├── uipath_storage_bucket.py   # Storage bucket access
│   └── uipath_context_grounding.py # Context grounding RAG
│
├── workflows/                      # Standalone workflows
│   └── web_search_resolution.py   # Web search workflow
│
├── config/
│   ├── config.py                  # Configuration settings
│   └── prompts.py                 # LLM prompt templates
│
└── utils/                         # Helper utilities
    ├── llm_service.py             # LLM client
    ├── article_search.py          # Article search utilities
    └── keyword_extraction.py      # Keyword extraction

tests/                              # Integration tests
docs/                               # Documentation
```

---

## ⚙️ Configuration

### Key Configuration Settings

**Knowledge Sufficiency** (`config.py`):
```python
KNOWLEDGE_SUFFICIENCY_THRESHOLD = 0.8  # Trigger web search if ≤ this score
```

**Article Semantic Re-Ranking** (NEW):
```python
ARTICLE_RERANK_TOP_K = 5           # Max articles after re-ranking
ARTICLE_RERANK_MIN_SCORE = 0.7     # Minimum relevance score
```

**Web Search**:
```python
WEB_SEARCH_ENABLED = True
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_TRUSTED_DOMAINS = [...]  # Whitelist of trusted domains
WEB_SEARCH_MIN_CONFIDENCE = 0.6
```

**Augmentation**:
- Max iterations: 2 (prevents infinite loops)
- Quality threshold: 0.5 (minimum score for augmented results)

---

## 🧪 Testing

### Test Individual Integrations
```bash
# Test ticket retrieval
python tests/test_uipath_get_job_data.py --ticket_id 12345

# Test storage bucket
python tests/test_uipath_storage_bucket.py

# Test context grounding
python tests/test_uipath_context_grounding.py --query "How to reset password?"
```

### Test Complete Workflow
```bash
# Run with real ticket
uv run uipath run agent '{"ticket_id": "42924"}'
```

### Expected Behavior (Slack VPN Example)
**Ticket**: "Can't access Slack on laptop after PTO"

**Expected Flow**:
1. ✓ Fetches ticket from FreshDesk
2. ✓ No IT action match (technical troubleshooting, not admin task)
3. ✓ Searches memory + Context Grounding
4. ✓ Fetches 10 FreshDesk articles via keywords
5. ✓ **Semantically re-ranks to top 5 relevant articles**
6. ✓ Evaluates sufficiency (likely insufficient on first pass)
7. ✓ Triggers web search (finds VPN requirement)
8. ✓ Extracts topics: ["Slack Desktop", "Cato VPN"]
9. ✓ Augments KB with "Slack Desktop Cato VPN configuration"
10. ✓ **Detects self-service viability** (user can connect VPN)
11. ✓ Generates **`ticket_response`** with clear VPN steps

**Response Type**: `ticket_response` (client-facing self-service)

**Sample Response**:
```
Hi Amelia,

I understand you're having trouble accessing Slack on your laptop after returning from PTO.
This is usually related to VPN connectivity.

Please try these steps:

1. Open the Cato Client VPN on your laptop and ensure you're connected
2. Once VPN is connected, close Slack Desktop completely (check system tray)
3. Reopen Slack Desktop - it should prompt you to sign in via SSO
4. Complete the SSO login process

Slack Desktop requires an active Cato VPN connection to authenticate. Since you can access
it on mobile (which uses app-based auth), this confirms your account is working fine.

If these steps don't resolve the issue, please reply and I'll investigate further.

Thank you!
```

---

## 📊 Monitoring & Logging

### Log Levels
Configured in `config.py`:
```python
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_PATH = "logs/agent.log"
```

### Key Log Messages
- `[Article Re-rank]` - Semantic filtering results
- `[Knowledge Sufficiency]` - Score evaluation
- `[Web Search]` - External search triggers
- `[Augment]` - KB re-querying
- `[Self-Service Check]` - Viability decision

---

## 🔧 Troubleshooting

### Issue: `uv run uipath init` encoding error
**Solution**: Run directly in PowerShell (not through Bash/WSL) due to Rich library spinner character encoding

### Issue: Articles not being filtered
**Solution**: Check `ARTICLE_RERANK_MIN_SCORE` in config.py, verify LLM is scoring correctly

### Issue: Web search not triggering
**Solution**: Check `KNOWLEDGE_SUFFICIENCY_THRESHOLD` (should be 0.8), verify score calculation in logs

### Issue: Always generating investigation actions instead of self-service
**Solution**: Check `_check_self_service_viability()` logic, verify LLM decision criteria

### Issue: Infinite augmentation loop
**Solution**: Verify max iteration check in `check_missing_information.py` (should be 2)

---

## 📚 Documentation

- **[Workflow Architecture](docs/WORKFLOW_ARCHITECTURE.md)** - Detailed workflow documentation
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude Code
- **[Response Quality Guidelines](docs/response_quality_guidelines.md)** - Response style guide

---

## 🔄 Recent Updates (v2.0)

### ✨ New Features
1. **Semantic Article Re-Ranking** - LLM-based filtering of FreshDesk articles
2. **Self-Service Detection** - Automatic detection of self-serviceable issues
3. **Autonomous Re-Routing** - Dynamic switching between internal/external knowledge
4. **Gap Analysis** - Iterative knowledge augmentation (max 2 iterations)
5. **Multi-Source Integration** - Comprehensive knowledge synthesis

### 🔧 Breaking Changes
- `search_memory` now runs AFTER `check_it_actions` (only if no IT match)
- Article results now include `relevance_score` field
- Response generation logic changed: 3 response types instead of 2

### 📈 Performance Improvements
- Reduced article noise by 50% (10 → top 5 relevant)
- Faster response generation with self-service detection
- Better knowledge quality through iterative augmentation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Support

For issues, questions, or feature requests:
- Create an issue in the repository
- Contact the IT Support team
- Check documentation in `docs/`

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

- **UiPath** - For LangGraph SDK and Chat models
- **Anthropic** - For Claude AI models
- **DuckDuckGo** - For web search API

