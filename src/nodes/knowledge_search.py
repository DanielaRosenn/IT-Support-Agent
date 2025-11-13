"""
Step 5: Knowledge Search Pipeline

Searches FreshService articles and Context Grounding, then analyzes if we can draft a response
and whether it's a self-service solution or requires IT action.
"""

import logging
from typing import Dict, Any, List
from src.graph.state import GraphState
from src.integrations.uipath_fetch_articles import fetch_articles_from_uipath
from src.utils.article_search import search_articles_unified
from src.integrations.uipath_context_grounding import ContextGroundingIntegration
from src.utils.keyword_extraction import extract_keywords_for_knowledge_search
from src.utils.keyword_extraction import extract_keywords_with_llm
from src.utils.llm_service import get_llm_service
from src.utils.prompt_utils import (
    log_llm_response,
    parse_llm_json_response,
    build_ticket_context,
    build_analysis_prompt,
    PromptBuilder
)
from src.config import config
from src.utils.exceptions import RecoverableError, LLMError, IntegrationError
import json

logger = logging.getLogger(__name__)


async def knowledge_search(state: GraphState) -> dict:
    """
    Step 5: Knowledge Search Pipeline

    Performs comprehensive knowledge search and analysis:
    - 5a: Search FreshService Articles
    - 5b: Search Context Grounding (IT Docs)
    - 5c: LLM Analysis - Can we draft response? Self-service or IT action needed?

    Args:
        state: Current graph state containing ticket_info and memory_results

    Returns:
        dict: Updated state with freshservice_articles, context_grounding_results,
              and knowledge_sufficiency
    """
    ticket_info = state["ticket_info"]
    memory_results = state.get("memory_results", [])

    logger.info(f"[Step 5] Starting Knowledge Search Pipeline")
    logger.info(f"  - Category: {ticket_info.category}")
    logger.info(f"  - Subject: {ticket_info.subject}")

    try:
        # [Step 5a] Search FreshService Articles
        logger.info("[Step 5a] Searching FreshService Articles...")

        try:
            # Extract keywords for article search
            keywords = await extract_keywords_with_llm(
                description=ticket_info.description,
                subject=ticket_info.subject,
                category=ticket_info.category
            )
            logger.info(f"[Step 5a] Extracted keywords: '{keywords}'")

            # Search articles using improved local cache (with UiPath fallback)
            articles_list = await search_articles_unified(
                keywords=keywords,
                limit=10,
                use_cache=True,  # Use local cache with improved scoring
                cache_fallback_to_uipath=True  # Fall back to UiPath if cache unavailable
            )

            logger.info(f"[Step 5a] Found {len(articles_list)} FreshService articles (before semantic filtering)")

            # Convert to dicts
            articles_raw = [
                {
                    "id": article.id,
                    "title": article.title,
                    "description": article.description,
                    "created_at": article.created_at,
                    "updated_at": article.updated_at
                }
                for article in articles_list
            ]

            # [Step 5a.1] Semantic Re-ranking - Filter by relevance
            if articles_raw:
                logger.info("[Step 5a.1] Performing semantic re-ranking of articles...")
                freshservice_articles = await _rerank_articles_by_relevance(
                    ticket_info=ticket_info,
                    articles=articles_raw
                )
                logger.info(f"[Step 5a.1] Filtered to {len(freshservice_articles)} articles after semantic re-ranking")
            else:
                freshservice_articles = []

            logger.info(f"[Step 5a] Final article count: {len(freshservice_articles)}")
            if freshservice_articles:
                for i, article in enumerate(freshservice_articles[:3], 1):
                    logger.info(f"  [{i}] {article.get('title', 'No title')} (ID: {article.get('id', 'N/A')})")

        except (LLMError, IntegrationError) as e:
            # Article search failed - continue without articles
            logger.warning(
                f"[Step 5a] Article search failed (recoverable): {e.message}",
                extra={"details": e.details, "ticket_id": ticket_info.ticket_id}
            )
            freshservice_articles = []

        # [Step 5b] Search Context Grounding (IT Docs)
        logger.info("[Step 5b] Searching Context Grounding IT Docs...")

        try:
            # Generate optimized search query for knowledge base
            search_query = await extract_keywords_for_knowledge_search(
                category=ticket_info.category,
                subject=ticket_info.subject,
                description=ticket_info.description
            )
            logger.info(f"[Step 5b] Generated Knowledge Base search query: '{search_query}'")

            # Search Context Grounding with refined query
            context_grounding = ContextGroundingIntegration()
            cg_results = context_grounding.search(
                query=search_query,
                number_of_results=5
            )

            logger.info(f"[Step 5b] Found {len(cg_results)} Context Grounding results")
            if cg_results:
                for i, result in enumerate(cg_results[:3], 1):
                    logger.info(f"  [{i}] Score: {result.score:.3f} - {result.source or 'No source'}")

            # Convert to dicts for state storage
            context_grounding_results = [
                {
                    "content": r.content,
                    "source": r.source,
                    "score": r.score,
                    "metadata": r.metadata
                }
                for r in cg_results
            ]

        except Exception as e:
            # Context grounding failed - continue without it
            logger.warning(
                f"[Step 5b] Context Grounding search failed (recoverable): {str(e)}",
                extra={"ticket_id": ticket_info.ticket_id}
            )
            context_grounding_results = []

        # [Step 5c] LLM Analysis - Can we draft response? Self-service or IT action?
        logger.info("[Step 5c] Analyzing knowledge sufficiency with LLM...")

        sufficiency_analysis = await _analyze_knowledge_sufficiency(
            ticket_info=ticket_info,
            memory_results=memory_results,
            freshservice_articles=freshservice_articles,
            context_grounding_results=context_grounding_results
        )

        logger.info(f"[Step 5c] Knowledge Sufficiency Analysis:")
        logger.info(f"  - Can Draft Response: {sufficiency_analysis['can_draft_response']}")
        logger.info(f"  - Confidence: {sufficiency_analysis['confidence']:.2f}")
        logger.info(f"  - Response Type: {sufficiency_analysis.get('response_type', 'unknown')}")
        logger.info(f"  - Reasoning: {sufficiency_analysis['reasoning'][:150]}...")

        return {
            "freshservice_articles": freshservice_articles,
            "context_grounding_results": context_grounding_results,
            "knowledge_sufficiency": sufficiency_analysis
        }

    except LLMError as e:
        # LLM analysis failed - return safe default
        logger.warning(
            f"[Step 5] LLM analysis failed (recoverable): {e.message}",
            extra={"details": e.details, "ticket_id": ticket_info.ticket_id}
        )
        return {
            "freshservice_articles": freshservice_articles if 'freshservice_articles' in locals() else [],
            "context_grounding_results": context_grounding_results if 'context_grounding_results' in locals() else [],
            "knowledge_sufficiency": {
                "can_draft_response": False,
                "confidence": 0.0,
                "response_type": "it_investigation",
                "reasoning": f"Knowledge analysis unavailable: {e.message}"
            }
        }

    except Exception as e:
        # Unexpected error - log and return safe default
        logger.error(
            f"[Step 5] Unexpected error in knowledge search: {str(e)}",
            exc_info=True,
            extra={"ticket_id": ticket_info.ticket_id}
        )
        return {
            "freshservice_articles": [],
            "context_grounding_results": [],
            "knowledge_sufficiency": {
                "can_draft_response": False,
                "confidence": 0.0,
                "response_type": "it_investigation",
                "reasoning": f"Knowledge search failed unexpectedly: {str(e)}"
            }
        }


async def _analyze_knowledge_sufficiency(
    ticket_info: Any,
    memory_results: List[Dict],
    freshservice_articles: List[Dict],
    context_grounding_results: List[Dict]
) -> Dict[str, Any]:
    """
    Use LLM to analyze if we have enough information to draft a response,
    and whether it's a self-service solution or requires IT action.

    Args:
        ticket_info: TicketInfo object
        memory_results: Results from Confluence Memory search
        freshservice_articles: Articles from FreshService
        context_grounding_results: Results from Context Grounding

    Returns:
        Dict with can_draft_response, confidence, response_type, reasoning, identified_sources
    """
    llm_service = get_llm_service()

    # Format all sources for the prompt
    sources_summary = _format_sources_for_analysis(
        memory_results,
        freshservice_articles,
        context_grounding_results
    )

    # Build ticket context
    ticket_context = build_ticket_context(ticket_info)

    # Analysis instructions
    analysis_instructions = """Evaluate if we can draft a high-quality response based on the available sources.

CRITICAL QUESTIONS TO ANSWER:
1. **Completeness**: Do the sources provide enough information to fully resolve this ticket?
2. **Accuracy**: Are the sources reliable and specific to this issue?
3. **Self-Service vs IT Action**: Can the USER execute the solution themselves, or does IT need to perform actions?

RESPONSE TYPE CLASSIFICATION:
- **self_service**: User can resolve this themselves (e.g., password reset via SSPR, following documented steps, self-help procedures)
- **it_execution**: Clear solution exists but IT must execute it (e.g., grant permissions, modify configurations, run admin commands)
- **it_investigation**: Information insufficient or unclear, IT must investigate further

CONFIDENCE GUIDELINES:
- **High (0.8-1.0)**: Multiple sources confirm same solution, clear actionable steps, no ambiguity
- **Medium (0.6-0.79)**: Solution exists but some details missing, or requires interpretation
- **Low (0.0-0.59)**: Information incomplete, conflicting sources, or no clear solution

RULES:
- If sources show clear self-service steps (like SSPR portal, documented user procedures) → response_type: "self_service"
- If solution requires admin privileges, system access, or IT tools → response_type: "it_execution"
- If information is vague, incomplete, or conflicting → response_type: "it_investigation"
- Set can_draft_response=true only if confidence >= 0.70

Respond with ONLY valid JSON:

{
    "can_draft_response": true or false,
    "confidence": 0.0 to 1.0,
    "response_type": "self_service" or "it_execution" or "it_investigation",
    "reasoning": "Detailed explanation of why you can/cannot draft response, which sources were most helpful, what's missing (if anything), and WHY you classified it as self-service/it-execution/investigation",
    "identified_sources": [
        {
            "type": "memory|article|context",
            "source_id": "source identifier",
            "relevance": 0.0 to 1.0,
            "provides": "what this source contributes to the solution"
        }
    ]
}

EXAMPLES:

SELF-SERVICE Example:
{
    "can_draft_response": true,
    "confidence": 0.85,
    "response_type": "self_service",
    "reasoning": "Multiple sources (Memory ticket #35768, FreshService Article #123) provide clear SSPR password reset instructions that user can follow themselves. Steps are documented and user-executable without admin access.",
    "identified_sources": [
        {"type": "memory", "source_id": "ticket_35768", "relevance": 0.9, "provides": "SSPR self-service password reset procedure"},
        {"type": "article", "source_id": "123", "relevance": 0.85, "provides": "Step-by-step SSPR portal access guide"}
    ]
}

IT-EXECUTION Example:
{
    "can_draft_response": true,
    "confidence": 0.75,
    "response_type": "it_execution",
    "reasoning": "Solution is clear from Context Grounding docs: user needs permissions added to AWS account. However, this requires admin access that only IT has. User cannot execute this themselves.",
    "identified_sources": [
        {"type": "context", "source_id": "aws_permissions_doc", "relevance": 0.8, "provides": "IAM permission assignment procedure for IT admins"}
    ]
}

IT-INVESTIGATION Example:
{
    "can_draft_response": false,
    "confidence": 0.45,
    "response_type": "it_investigation",
    "reasoning": "No sources directly address this specific VPN connection issue. Context Grounding has general VPN troubleshooting but nothing matching these symptoms. Needs IT investigation to diagnose root cause.",
    "identified_sources": [
        {"type": "context", "source_id": "vpn_general", "relevance": 0.4, "provides": "General VPN setup guide, not specific to this error"}
    ]
}"""

    # Build the prompt using the builder
    prompt = build_analysis_prompt(
        task_title="KNOWLEDGE SUFFICIENCY ANALYSIS",
        ticket_context=ticket_context,
        available_info=sources_summary,
        analysis_instructions=analysis_instructions
    )

    logger.debug("Sending knowledge sufficiency analysis to LLM")

    response = await llm_service.invoke(
        prompt=prompt,
        max_tokens=800,
        temperature=0.2  # Slightly higher for nuanced analysis
    )

    # Log and parse response
    log_llm_response(logger, response, "SUFFICIENCY ANALYSIS RESPONSE")

    try:
        result = parse_llm_json_response(response, logger, "knowledge sufficiency analysis")

        # Validate required fields
        required = ["can_draft_response", "confidence", "response_type", "reasoning"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate response_type
        valid_types = ["self_service", "it_execution", "it_investigation"]
        if result["response_type"] not in valid_types:
            logger.warning(f"Invalid response_type '{result['response_type']}', defaulting to 'it_investigation'")
            result["response_type"] = "it_investigation"

        # Enforce confidence threshold
        if result["can_draft_response"] and result["confidence"] < 0.70:
            logger.warning(f"Confidence {result['confidence']} too low, setting can_draft_response=False")
            result["can_draft_response"] = False

        return result

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse LLM sufficiency analysis: {e}")
        logger.error(f"Raw response: {response}")
        # Return safe default
        return {
            "can_draft_response": False,
            "confidence": 0.0,
            "response_type": "it_investigation",
            "reasoning": f"Failed to analyze knowledge sufficiency: {str(e)}",
            "identified_sources": []
        }


def _format_sources_for_analysis(
    memory_results: List[Dict],
    freshservice_articles: List[Dict],
    context_grounding_results: List[Dict]
) -> str:
    """Format all sources into a readable summary for LLM analysis."""

    sections = []

    # Confluence Memory Results
    if memory_results:
        memory_section = "CONFLUENCE MEMORY (Previously Resolved Tickets):\n"
        for i, result in enumerate(memory_results[:3], 1):
            memory_section += f"[{i}] Score: {result.get('score', 0):.3f}\n"
            memory_section += f"    Source: {result.get('source', 'Unknown')}\n"
            memory_section += f"    Content: {result.get('content', '')[:300]}...\n\n"
        sections.append(memory_section)
    else:
        sections.append("CONFLUENCE MEMORY: No similar resolved tickets found\n")

    # FreshService Articles
    if freshservice_articles:
        articles_section = "FRESHSERVICE ARTICLES:\n"
        for i, article in enumerate(freshservice_articles[:5], 1):
            articles_section += f"[{i}] Article ID: {article.get('id', 'N/A')}\n"
            articles_section += f"    Title: {article.get('title', 'No title')}\n"
            articles_section += f"    Description: {article.get('description', '')[:200]}...\n\n"
        sections.append(articles_section)
    else:
        sections.append("FRESHSERVICE ARTICLES: No relevant articles found\n")

    # Context Grounding Results
    if context_grounding_results:
        cg_section = "CONTEXT GROUNDING (IT Documentation):\n"
        for i, result in enumerate(context_grounding_results[:5], 1):
            cg_section += f"[{i}] Score: {result.get('score', 0):.3f}\n"
            cg_section += f"    Source: {result.get('source', 'Unknown')}\n"
            cg_section += f"    Content: {result.get('content', '')[:300]}...\n\n"
        sections.append(cg_section)
    else:
        sections.append("CONTEXT GROUNDING: No relevant documentation found\n")

    return "\n".join(sections)


async def _rerank_articles_by_relevance(
    ticket_info: Any,
    articles: List[Dict]
) -> List[Dict]:
    """
    Use LLM to semantically re-rank articles based on ticket relevance.

    Handles large article sets by batching to prevent LLM response truncation.

    Logic:
    - If ≤5 articles: Score all, keep only those with score ≥ 0.7
    - If >5 articles: Score all, sort by score, keep top 5

    Args:
        ticket_info: TicketInfo object with ticket details
        articles: List of article dicts with id, title, description

    Returns:
        List of filtered articles (with relevance_score added)
    """
    if not articles:
        return []

    llm_service = get_llm_service()
    num_articles = len(articles)

    logger.info(f"[Article Re-rank] Starting re-ranking for {num_articles} articles")

    # Batch processing to prevent LLM response truncation
    BATCH_SIZE = 10  # Process max 10 articles per LLM call

    if num_articles > BATCH_SIZE:
        logger.info(f"[Article Re-rank] Using batch processing ({BATCH_SIZE} articles per batch)")
        return await _rerank_articles_in_batches(ticket_info, articles, llm_service, BATCH_SIZE)

    # For small sets, process all at once
    logger.info(f"[Article Re-rank] Processing {num_articles} articles in single batch")
    scored_articles = await _rerank_single_batch(ticket_info, articles, llm_service)

    # Log score assignment
    logger.info(f"[Article Re-rank] Score assignment complete:")
    for article in scored_articles:
        logger.info(f"[Article Re-rank]   ID {article['id']}: score={article.get('relevance_score', 'NOT SET')}")

    # Apply filtering logic
    quality_articles = [
        a for a in scored_articles
        if a["relevance_score"] >= config.ARTICLE_RERANK_MIN_SCORE
    ]

    # If more than TOP_K, keep only top K
    if len(quality_articles) > config.ARTICLE_RERANK_TOP_K:
        articles_sorted = sorted(quality_articles, key=lambda x: x["relevance_score"], reverse=True)
        filtered = articles_sorted[:config.ARTICLE_RERANK_TOP_K]
        logger.info(
            f"[Article Re-rank] {num_articles} articles → "
            f"{len(quality_articles)} above threshold (≥{config.ARTICLE_RERANK_MIN_SCORE}) → "
            f"kept top {len(filtered)}"
        )
    else:
        filtered = quality_articles
        logger.info(
            f"[Article Re-rank] {num_articles} articles → "
            f"kept {len(filtered)} with score ≥ {config.ARTICLE_RERANK_MIN_SCORE}"
        )

    # Log filtered articles with scores
    for i, article in enumerate(filtered, 1):
        logger.info(
            f"[Article Re-rank]   [{i}] Score: {article['relevance_score']:.2f} | "
            f"ID: {article['id']} | Title: {article['title'][:50]}..."
        )

    return filtered


async def _rerank_articles_in_batches(
    ticket_info: Any,
    articles: List[Dict],
    llm_service: Any,
    batch_size: int
) -> List[Dict]:
    """
    Re-rank articles in batches to prevent LLM response truncation.

    Args:
        ticket_info: TicketInfo object
        articles: List of all articles
        llm_service: LLM service instance
        batch_size: Number of articles per batch

    Returns:
        List of all articles with relevance_score added
    """
    num_articles = len(articles)
    num_batches = (num_articles + batch_size - 1) // batch_size  # Ceiling division

    logger.info(f"[Article Re-rank] Processing {num_articles} articles in {num_batches} batches")

    all_scored_articles = []

    # Process each batch
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, num_articles)
        batch = articles[start_idx:end_idx]

        logger.info(f"[Article Re-rank] Batch {batch_num + 1}/{num_batches}: Processing articles {start_idx + 1}-{end_idx}")

        try:
            scored_batch = await _rerank_single_batch(ticket_info, batch, llm_service)
            all_scored_articles.extend(scored_batch)
            logger.info(f"[Article Re-rank] Batch {batch_num + 1}/{num_batches}: Scored {len(scored_batch)} articles")
        except Exception as e:
            logger.error(f"[Article Re-rank] Batch {batch_num + 1}/{num_batches} failed: {e}")
            # On batch failure, assign default score of 0 to this batch
            for article in batch:
                article["relevance_score"] = 0.0
            all_scored_articles.extend(batch)
            logger.warning(f"[Article Re-rank] Batch {batch_num + 1}/{num_batches}: Assigned default scores due to error")

    # Now apply filtering logic to all scored articles
    num_articles = len(all_scored_articles)

    # Log score assignment
    logger.info(f"[Article Re-rank] Score assignment complete for all batches:")
    for article in all_scored_articles:
        logger.info(f"[Article Re-rank]   ID {article['id']}: score={article.get('relevance_score', 'NOT SET')}")

    # Apply filtering logic: ALWAYS enforce minimum score threshold
    quality_articles = [
        a for a in all_scored_articles
        if a["relevance_score"] >= config.ARTICLE_RERANK_MIN_SCORE
    ]

    # If more than TOP_K, keep only top K
    if len(quality_articles) > config.ARTICLE_RERANK_TOP_K:
        articles_sorted = sorted(quality_articles, key=lambda x: x["relevance_score"], reverse=True)
        filtered = articles_sorted[:config.ARTICLE_RERANK_TOP_K]
        logger.info(
            f"[Article Re-rank] {num_articles} articles → "
            f"{len(quality_articles)} above threshold (≥{config.ARTICLE_RERANK_MIN_SCORE}) → "
            f"kept top {len(filtered)}"
        )
    else:
        filtered = quality_articles
        logger.info(
            f"[Article Re-rank] {num_articles} articles → "
            f"kept {len(filtered)} with score ≥ {config.ARTICLE_RERANK_MIN_SCORE}"
        )

    # Log filtered articles with scores
    for i, article in enumerate(filtered, 1):
        logger.info(
            f"[Article Re-rank]   [{i}] Score: {article['relevance_score']:.2f} | "
            f"ID: {article['id']} | Title: {article['title'][:50]}..."
        )

    return filtered


async def _rerank_single_batch(
    ticket_info: Any,
    articles: List[Dict],
    llm_service: Any
) -> List[Dict]:
    """
    Re-rank a single batch of articles using LLM.

    Args:
        ticket_info: TicketInfo object
        articles: List of articles to score (≤10 recommended)
        llm_service: LLM service instance

    Returns:
        List of articles with relevance_score added
    """
    num_articles = len(articles)

    # Format articles for prompt
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += f"[{i}] ID: {article['id']}\n"
        articles_text += f"    Title: {article['title']}\n"
        articles_text += f"    Description: {article['description'][:300]}...\n\n"

    # Format ticket info
    ticket_section = f"""SUBJECT: {ticket_info.subject}
CATEGORY: {ticket_info.category}
DESCRIPTION: {ticket_info.description}"""

    # Scoring instructions
    scoring_instructions = f"""For each article, assign a relevance score (0.0-1.0) based on how well it addresses this specific ticket.

SCORING CRITERIA:

**0.9-1.0**: Direct match - Article directly solves the exact problem described in the ticket
**0.7-0.89**: Strong relevance - Article addresses the same system/application with relevant procedures
**0.5-0.69**: Moderate relevance - Related topic, may provide useful context
**0.3-0.49**: Weak relevance - Tangentially related, limited usefulness
**0.0-0.29**: Not relevant - Wrong topic or category

EVALUATION FACTORS:
- Does the article title/description match the ticket's subject?
- Does the article address the specific issue mentioned?
- Would this article help resolve the ticket?
- Is the article specific enough to be actionable?

Return ONLY valid JSON with scores for ALL articles:

{{
  "scores": [
    {{"article_id": {articles[0]['id']}, "score": 0.95, "reasoning": "Direct match - addresses exact issue"}},
    {{"article_id": {articles[1]['id'] if len(articles) > 1 else 'N/A'}, "score": 0.65, "reasoning": "Related topic"}},
    ...
  ]
}}

IMPORTANT: Include ALL {num_articles} articles in your response."""

    # Build prompt using PromptBuilder
    prompt = (PromptBuilder("ARTICLE RELEVANCE SCORING", include_guidelines=False)
        .add_section("TICKET INFORMATION", ticket_section)
        .add_section(f"ARTICLES TO RANK ({num_articles} total)", articles_text)
        .add_section("YOUR TASK: SCORE EACH ARTICLE'S RELEVANCE", scoring_instructions)
        .build())

    try:
        logger.debug("[Article Re-rank] Sending re-ranking request to LLM")

        response = await llm_service.invoke(
            prompt=prompt,
            max_tokens=800,
            temperature=0.2
        )

        log_llm_response(logger, response, "ARTICLE RE-RANKING")

        # Parse JSON response
        result = parse_llm_json_response(response, logger, "article re-ranking")

        if "scores" not in result:
            logger.warning("[Article Re-rank] No 'scores' field in LLM response, returning all articles")
            return articles

        # Convert article_ids to strings to handle JSON parsing (LLM may return integers)
        scores_dict = {str(item["article_id"]): item["score"] for item in result["scores"]}

        # Add scores to articles
        for article in articles:
            article_id = str(article["id"])  # Ensure string comparison
            article["relevance_score"] = scores_dict.get(article_id, 0.0)

        # Return scored articles (filtering happens in caller)
        return articles

    except Exception as e:
        logger.error(f"[Article Re-rank] Failed to re-rank batch: {str(e)}", exc_info=True)
        # On error, assign default score of 0 to all articles
        for article in articles:
            article["relevance_score"] = 0.0
        return articles
