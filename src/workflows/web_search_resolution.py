"""
Web Search Resolution Workflow
Generates ticket resolutions using web search + Claude (Bedrock)

Flow:
1. Claude: Build optimized search query from ticket
2. DuckDuckGo: Search with whitelist filtering
3. Claude: Rank and filter results by relevance
4. Claude: Generate structured resolution from top results
"""

import logging
import os
from typing import List, Optional
from pydantic import BaseModel, Field

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

from src.integrations.uipath_get_job_data import TicketInfo
from src.integrations.web_search import WebSearchIntegration, SearchResult
from src.config import config

logger = logging.getLogger(__name__)


class RankedResult(BaseModel):
    """Search result with relevance ranking"""
    result: SearchResult
    relevance_score: float = Field(description="Relevance score 0-1")
    reasoning: str = Field(description="Why this result is relevant")


class WebSearchResolution(BaseModel):
    """Resolution generated from web search"""
    resolution_steps: List[str] = Field(description="Numbered action steps")
    sources: List[str] = Field(description="URLs used as sources")
    search_query_used: str = Field(description="Optimized search query")
    confidence: float = Field(description="Confidence score 0-1")


class WebSearchResolutionWorkflow:
    """Orchestrates web search resolution generation with Claude enhancement"""

    def __init__(self):
        """Initialize workflow with Claude (Bedrock) and web search"""
        # Initialize Bedrock client
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        # Use ChatBedrockConverse for direct Bedrock access with LangChain tracing
        self.llm = ChatBedrockConverse(
            model=config.BEDROCK_MODEL_ID,
            client=bedrock_client,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        self.web_search = WebSearchIntegration()
        logger.info(f"WebSearchResolution workflow initialized with Bedrock ({config.BEDROCK_MODEL_ID})")

    async def generate_resolution(
        self,
        ticket: TicketInfo,
        max_search_results: int = None
    ) -> WebSearchResolution:
        """
        Generate resolution using web search + Claude

        Args:
            ticket: Ticket information
            max_search_results: Max search results (uses config if None)

        Returns:
            WebSearchResolution with steps and sources
        """
        max_search_results = max_search_results or config.WEB_SEARCH_MAX_RESULTS

        logger.info(f"Starting web search resolution for ticket: {ticket.ticket_id}")

        # Step 1: Build optimized search query
        logger.info("[Step 1/4] Building optimized search query with Claude")
        search_query = await self._build_search_query(ticket)
        logger.info(f"✓ Optimized query: '{search_query}'")

        # Step 2: Perform web search
        logger.info(f"[Step 2/4] Searching DuckDuckGo with whitelist filtering")
        search_results = await self.web_search.search(search_query, max_search_results)
        logger.info(f"✓ Found {len(search_results)} filtered results")

        if not search_results:
            logger.warning("No search results found from trusted domains")
            return WebSearchResolution(
                resolution_steps=["No relevant web resources found from trusted sources."],
                sources=[],
                search_query_used=search_query,
                confidence=0.0
            )

        # Step 3: Rank and filter results
        logger.info(f"[Step 3/4] Ranking results by relevance with Claude")
        ranked_results = await self._rank_results(ticket, search_results)
        top_results = ranked_results[:config.WEB_SEARCH_TOP_RESULTS_TO_USE]
        logger.info(f"✓ Selected top {len(top_results)} results")

        # Step 4: Generate resolution
        logger.info(f"[Step 4/4] Generating structured resolution with Claude")
        resolution = await self._generate_resolution(ticket, top_results, search_query)
        logger.info(f"✓ Resolution generated (confidence: {resolution.confidence:.2f})")

        return resolution

    async def _build_search_query(self, ticket: TicketInfo) -> str:
        """
        Step 1: Use Claude to build optimized search query

        Args:
            ticket: Ticket information

        Returns:
            Optimized search query string
        """
        system_prompt = """You are an IT support search query optimizer. Your task is to create an optimal web search query to find solutions for IT support tickets.

Guidelines:
- Focus on technical terms and specific symptoms
- Include relevant technology names (Windows, Active Directory, VPN, etc.)
- Remove conversational language and typos
- Keep query concise (3-8 words)
- Focus on actionable solutions, not just explanations

Output only the search query, nothing else."""

        user_prompt = f"""Create an optimized search query for this IT support ticket:

Subject: {ticket.subject}
Category: {ticket.category}
Description: {ticket.description}

Search query:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            query = response.content.strip()

            # Clean up query (remove quotes, extra whitespace)
            query = query.replace('"', '').replace("'", '')
            query = ' '.join(query.split())

            logger.debug(f"Generated search query: {query}")
            return query

        except Exception as e:
            logger.error(f"Failed to build search query: {e}")
            # Fallback: use subject as query
            return ticket.subject

    async def _rank_results(
        self,
        ticket: TicketInfo,
        results: List[SearchResult]
    ) -> List[RankedResult]:
        """
        Step 3: Use Claude to rank results by relevance

        Args:
            ticket: Ticket information
            results: Search results to rank

        Returns:
            List of RankedResult sorted by relevance score
        """
        system_prompt = """You are an IT support result evaluator. Analyze search results and rate their relevance to the ticket.

For each result, provide:
1. Relevance score (0.0 to 1.0)
2. Brief reasoning (one sentence)

Focus on:
- Technical accuracy and specificity
- Actionable steps provided
- Relevance to the exact issue described
- Source credibility (official docs score higher)"""

        # Build results summary
        results_text = ""
        for idx, result in enumerate(results, 1):
            results_text += f"\nResult {idx}:\n"
            results_text += f"Title: {result.title}\n"
            results_text += f"Source: {result.source_domain}\n"
            results_text += f"Content: {result.content}\n"

        user_prompt = f"""Ticket Information:
Subject: {ticket.subject}
Category: {ticket.category}
Description: {ticket.description}

Search Results to Rank:
{results_text}

For each result, output in this exact format:
Result X: [score] - [reasoning]

Example:
Result 1: 0.9 - Official Microsoft docs with exact steps for password reset
Result 2: 0.4 - Generic tutorial, not specific to Active Directory

Now rank all {len(results)} results:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            rankings_text = response.content.strip()

            # Parse rankings
            ranked_results = self._parse_rankings(rankings_text, results)

            # Sort by score (descending)
            ranked_results.sort(key=lambda x: x.relevance_score, reverse=True)

            logger.debug(f"Ranked {len(ranked_results)} results")
            for idx, ranked in enumerate(ranked_results[:3], 1):
                logger.debug(f"  #{idx}: {ranked.relevance_score:.2f} - {ranked.result.title[:50]}")

            return ranked_results

        except Exception as e:
            logger.error(f"Failed to rank results: {e}")
            # Fallback: return all with default score
            return [RankedResult(
                result=r,
                relevance_score=0.5,
                reasoning="Default ranking (Claude ranking failed)"
            ) for r in results]

    def _parse_rankings(
        self,
        rankings_text: str,
        results: List[SearchResult]
    ) -> List[RankedResult]:
        """
        Parse Claude's ranking output

        Args:
            rankings_text: Claude's response
            results: Original results

        Returns:
            List of RankedResult
        """
        import re

        ranked = []
        lines = rankings_text.split('\n')

        for line in lines:
            # Match pattern: "Result X: 0.X - reasoning"
            match = re.match(r'Result\s+(\d+):\s*([\d.]+)\s*-\s*(.+)', line.strip())
            if match:
                idx = int(match.group(1)) - 1  # Convert to 0-based
                score = float(match.group(2))
                reasoning = match.group(3).strip()

                if 0 <= idx < len(results):
                    ranked.append(RankedResult(
                        result=results[idx],
                        relevance_score=min(max(score, 0.0), 1.0),  # Clamp 0-1
                        reasoning=reasoning
                    ))

        # If parsing failed, return all with default scores
        if not ranked:
            logger.warning("Failed to parse rankings, using default scores")
            return [RankedResult(
                result=r,
                relevance_score=0.5,
                reasoning="Default ranking"
            ) for r in results]

        return ranked

    async def _generate_resolution(
        self,
        ticket: TicketInfo,
        ranked_results: List[RankedResult],
        search_query: str
    ) -> WebSearchResolution:
        """
        Step 4: Generate structured resolution from top results

        Args:
            ticket: Ticket information
            ranked_results: Top-ranked search results
            search_query: Search query used

        Returns:
            WebSearchResolution with steps and sources
        """
        system_prompt = """You are an IT support resolution generator. Create clear, actionable step-by-step resolutions based on web search results.

Guidelines:
- Provide numbered, specific action steps
- Focus on the exact issue in the ticket
- Be concise but complete
- Cite specific sources when referencing their steps
- Include verification steps when appropriate
- Provide 3-10 steps (depending on complexity)

Output format:
1. First step
2. Second step
3. Third step
etc.

Then on a new line, add:
CONFIDENCE: [0.0-1.0]"""

        # Build context from top results
        context_text = ""
        sources = []
        for idx, ranked in enumerate(ranked_results, 1):
            result = ranked.result
            context_text += f"\n--- Source {idx} ({result.source_domain}) ---\n"
            context_text += f"Title: {result.title}\n"
            context_text += f"Content: {result.content}\n"
            context_text += f"Relevance: {ranked.relevance_score:.2f} - {ranked.reasoning}\n"
            sources.append(result.url)

        user_prompt = f"""Ticket Information:
Subject: {ticket.subject}
Category: {ticket.category}
Description: {ticket.description}
Requester: {ticket.requester}

Web Search Results:
{context_text}

Generate a step-by-step resolution for this ticket based on the search results above:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            resolution_text = response.content.strip()

            # Parse resolution steps and confidence
            steps, confidence = self._parse_resolution(resolution_text)

            return WebSearchResolution(
                resolution_steps=steps,
                sources=sources,
                search_query_used=search_query,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Failed to generate resolution: {e}")
            return WebSearchResolution(
                resolution_steps=["Error generating resolution from web search results."],
                sources=sources,
                search_query_used=search_query,
                confidence=0.0
            )

    def _parse_resolution(self, resolution_text: str) -> tuple[List[str], float]:
        """
        Parse Claude's resolution output

        Args:
            resolution_text: Claude's response

        Returns:
            Tuple of (steps list, confidence float)
        """
        import re

        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', resolution_text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.7

        # Clamp confidence
        confidence = min(max(confidence, 0.0), 1.0)

        # Remove confidence line
        resolution_text = re.sub(r'CONFIDENCE:.*', '', resolution_text).strip()

        # Extract numbered steps
        steps = []
        lines = resolution_text.split('\n')

        for line in lines:
            line = line.strip()
            # Match numbered steps (1. 2. etc.)
            match = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if match:
                steps.append(match.group(1).strip())

        # If no numbered steps found, treat each line as a step
        if not steps:
            steps = [line.strip() for line in lines if line.strip()]

        return steps, confidence


# Convenience function
async def generate_resolution_from_web_search(
    ticket: TicketInfo,
    max_results: int = None
) -> WebSearchResolution:
    """
    Convenience function to generate resolution from web search

    Args:
        ticket: Ticket information
        max_results: Max search results (uses config if None)

    Returns:
        WebSearchResolution
    """
    workflow = WebSearchResolutionWorkflow()
    return await workflow.generate_resolution(ticket, max_results)