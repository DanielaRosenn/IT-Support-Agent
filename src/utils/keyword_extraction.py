"""
Keyword Extraction Utilities
Uses Claude (via AWS Bedrock) to extract keywords from ticket data for article search
"""

import logging
from typing import Optional
from src.utils.llm_service import get_llm_service

logger = logging.getLogger(__name__)


async def extract_keywords_with_llm(
    description: str,
    subject: str = "",
    category: str = "",
    max_keywords: int = 10
) -> str:
    """
    Extract relevant keywords from ticket information using Claude LLM

    Args:
        description: Ticket description
        subject: Ticket subject (optional)
        category: Ticket category (optional)
        max_keywords: Maximum number of keywords to extract (default: 10)

    Returns:
        Space-separated string of keywords

    Example usage:
        keywords = await extract_keywords_with_llm(
            "I need to reset my password for VPN access",
            "Password Reset Request"
        )
        # Returns: 'password reset vpn access'
    """
    try:
        # Get LLM service
        llm_service = get_llm_service()

        if not llm_service.is_available():
            logger.warning("LLM service not available, using fallback")
            return _fallback_keyword_extraction(description, subject, category)

        # Build the prompt
        prompt_text = _build_keyword_extraction_prompt(
            description=description,
            subject=subject,
            category=category,
            max_keywords=max_keywords
        )

        logger.debug(f"Sending keyword extraction request to Bedrock")

        # Get response from Bedrock
        response = await llm_service.invoke(
            prompt=prompt_text,
            max_tokens=200,
            temperature=0.1
        )

        # Extract keywords from response
        keywords = _parse_keyword_response(response)

        logger.info(f"Extracted keywords: {keywords}")
        return keywords

    except Exception as e:
        logger.error(f"Error extracting keywords with LLM: {e}", exc_info=True)
        # Fallback: use basic extraction from text
        return _fallback_keyword_extraction(description, subject, category)


def _build_keyword_extraction_prompt(
    description: str,
    subject: str,
    category: str,
    max_keywords: int
) -> str:
    """
    Build prompt for Claude to extract keywords

    Args:
        description: Ticket description
        subject: Ticket subject
        category: Ticket category
        max_keywords: Maximum keywords

    Returns:
        Formatted prompt string
    """
    prompt = f"""Extract 3-6 search keywords from this IT support ticket for finding knowledge base articles.

RULES:
- Include: System names (SSPR, NetSuite, 1Password, SFDC, Jump Host, JH2, VPN), action verbs (reset, unlock, access, setup), technical terms (password, account, login)
- Keep abbreviations exact: SSPR, SFDC, JH2, SSO - never expand or drop them
- Use "reset" not "change" for passwords
- Exclude: Person names, company names (catonetworks), generic words (need, help, issue, problem), pronouns (my, your, the)
- Output ONLY keywords, space-separated, lowercase

Ticket:"""

    if subject:
        prompt += f"\nSubject: {subject}"
    if category:
        prompt += f"\nCategory: {category}"
    if description:
        prompt += f"\nDescription: {description}"

    prompt += f"""

Examples:
"Need reset password for JH (Jump Host)" → password reset jump host
"Can't access email after password change" → password reset email access
"VPN not connecting on laptop" → vpn connection laptop
"Locked out of account, SSPR not working" → sspr password reset account locked

Keywords:"""

    return prompt


def _parse_keyword_response(response: str) -> str:
    """
    Parse Claude's response to extract keywords

    Args:
        response: Claude's response text

    Returns:
        Cleaned keyword string
    """
    # Remove any extra whitespace and newlines
    keywords = response.strip()

    # Remove any markdown formatting if present
    keywords = keywords.replace('`', '')
    keywords = keywords.replace('*', '')

    # Take only the first line if multiple lines returned
    keywords = keywords.split('\n')[0]

    # Normalize spaces
    keywords = ' '.join(keywords.split())

    return keywords.lower()


def _fallback_keyword_extraction(
    description: str,
    subject: str,
    category: str
) -> str:
    """
    Fallback keyword extraction if LLM fails
    Simple approach: take first N words from subject and description

    Args:
        description: Ticket description
        subject: Ticket subject
        category: Ticket category

    Returns:
        Space-separated keywords
    """
    logger.warning("Using fallback keyword extraction")

    # Combine text sources
    text_parts = []
    if subject:
        text_parts.append(subject)
    if category:
        text_parts.append(category)
    if description:
        # Take first 50 words of description
        words = description.split()[:50]
        text_parts.append(' '.join(words))

    combined = ' '.join(text_parts)

    # Basic cleanup
    import re
    combined = re.sub(r'[^a-zA-Z0-9\s]', ' ', combined)
    combined = ' '.join(combined.split())

    # Return first 100 characters as keywords
    return combined[:100].lower()


async def summarize_ticket_for_search(
    ticket_id: str,
    description: str,
    subject: str = "",
    category: str = "",
    **kwargs
) -> str:
    """
    Summarize ticket data into keywords suitable for article search using Claude LLM

    This is the main function to use when preparing ticket data for
    the article fetching UiPath process.

    Args:
        ticket_id: Ticket ID (for logging)
        description: Ticket description
        subject: Ticket subject
        category: Ticket category
        **kwargs: Additional ticket fields (ignored)

    Returns:
        Keyword string for article search

    Example usage:
        from src.integrations.uipath_get_job_data import fetch_ticket_from_uipath

        ticket = await fetch_ticket_from_uipath("12345")
        keywords = await summarize_ticket_for_search(**ticket.model_dump())
        # Use keywords for article fetching
    """
    logger.info(f"Summarizing ticket {ticket_id} for article search using Claude")

    keywords = await extract_keywords_with_llm(
        description=description,
        subject=subject,
        category=category
    )

    logger.info(f"Ticket {ticket_id} summarized to keywords: {keywords}")

    return keywords


def summarize_ticket_for_search_sync(
    ticket_id: str,
    description: str,
    subject: str = "",
    category: str = "",
    **kwargs
) -> str:
    """
    Synchronous wrapper for summarize_ticket_for_search

    Use this in non-async contexts (e.g., LangGraph nodes that aren't async)

    Args:
        ticket_id: Ticket ID (for logging)
        description: Ticket description
        subject: Ticket subject
        category: Ticket category
        **kwargs: Additional ticket fields (ignored)

    Returns:
        Keyword string for article search
    """
    import asyncio

    try:
        # Try to get running event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create new one
        return asyncio.run(
            summarize_ticket_for_search(
                ticket_id=ticket_id,
                description=description,
                subject=subject,
                category=category,
                **kwargs
            )
        )
    else:
        # Already in async context, shouldn't use this function
        raise RuntimeError(
            "summarize_ticket_for_search_sync() called from async context. "
            "Use 'await summarize_ticket_for_search()' instead."
        )


async def extract_keywords_for_confluence_memory(
    category: str,
    subject: str,
    description: str
) -> str:
    """
    Generate semantic search query for Confluence Memory (Context Grounding).

    Context Grounding uses semantic search with natural language questions,
    not keyword matching. We formulate a question that finds similar resolved tickets.

    Formula: "Based on [document type], [question about issue]. [What you need]"

    Args:
        category: Ticket category (e.g., "Software")
        subject: Ticket subject (e.g., "Reset Password")
        description: Ticket description/details

    Returns:
        Natural language search query for finding similar resolved tickets

    Examples:
        Input: category="Software", subject="Reset Password", description="User needs JH password reset"
        Output: "Based on previous IT support tickets, how do I reset a JH (Jump Host) password for a user?"

        Input: category="Network", subject="VPN Access", description="Cannot connect to corporate VPN"
        Output: "From past ticket resolutions, what are the solutions for VPN connection issues?"
    """
    try:
        llm_service = get_llm_service()

        if not llm_service.is_available():
            logger.warning("LLM service not available, using fallback for Confluence search")
            return _fallback_confluence_keyword_extraction(category, subject, description)

        # Build prompt for generating semantic search query
        prompt = f"""Convert this IT support ticket into a concise search query to find similar previously resolved tickets.

Ticket Category: {category}
Ticket Subject: {subject}
Ticket Description: {description}

RULES - Extract ONLY core technical symptoms:
1. Focus on: System/app name + technical symptom/error
2. Keep abbreviations exact: JH, SSPR, SFDC, VPN, SSO (never expand)
3. Include diagnostic patterns: "works on X but not Y", "cannot access", "connection failed"
4. EXCLUDE: Person names, dates, temporal context (after PTO, last week, yesterday), polite language (please, thank you), urgency words
5. Keep it SHORT: Maximum 10-12 words
6. Use simple present tense

Examples:

Ticket: "Hi, returned from vacation yesterday. Slack won't work on my laptop but my phone is fine. Please help!"
Query: Slack works on mobile but not on laptop

Ticket: "User Mary needs JH password reset ASAP"
Query: JH password reset

Ticket: "After updating, SSPR not working and account is locked"
Query: SSPR fails and account locked

Ticket: "Cannot access VPN on new laptop since Monday"
Query: VPN connection issues on laptop

Ticket: "Good morning team, Outlook email configuration giving me an error message. This is urgent."
Query: Outlook email configuration error

Ticket: "I was on PTO and now I can't login to SFDC. Works fine for my colleague."
Query: Cannot access SFDC

Search Query:"""

        logger.debug("Generating Confluence Memory search query")

        response = await llm_service.invoke(
            prompt=prompt,
            max_tokens=150,
            temperature=0.1  # Low temperature for consistent extraction
        )

        # Clean response
        query = response.strip()
        # Remove any quotes if LLM added them
        query = query.strip('"').strip("'")
        # Take only first sentence if multiple returned
        if '?' in query:
            query = query.split('?')[0] + '?'

        logger.info(f"Generated Confluence Memory search query: {query}")
        return query

    except Exception as e:
        logger.error(f"Error generating Confluence Memory search query: {e}", exc_info=True)
        return _fallback_confluence_keyword_extraction(category, subject, description)


async def extract_keywords_for_knowledge_search(
    category: str,
    subject: str,
    description: str
) -> str:
    """
    Generate optimized search query for Knowledge Base articles (Context Grounding IT documentation).

    Creates a concise query to find relevant troubleshooting guides, procedures,
    and IT documentation from Confluence knowledge base.

    Args:
        category: Ticket category (e.g., "Email", "Network", "Software")
        subject: Ticket subject (e.g., "Can't access email")
        description: Ticket description/details

    Returns:
        Optimized search query string (5-15 words) for finding IT documentation

    Examples:
        Input: category="Email", subject="Can't access email",
               description="Outlook won't open, profile cannot be loaded"
        Output: "Outlook profile cannot be loaded error troubleshooting"

        Input: category="Network", subject="VPN connection failed",
               description="Getting error 809 when connecting from home"
        Output: "VPN connection error 809 troubleshooting remote access"
    """
    try:
        llm_service = get_llm_service()

        if not llm_service.is_available():
            logger.warning("LLM service not available, using fallback for Confluence search")
            return _fallback_confluence_keyword_extraction(category, subject, description)

        # Build prompt for generating semantic search query
        prompt = f"""Extract a search query from this IT support ticket to find troubleshooting documentation.

Ticket Category: {category}
Ticket Subject: {subject}
Ticket Description: {description}

RULES:
1. Start with SUBJECT as primary source, use DESCRIPTION for error codes and technical details
2. Keep system names EXACT (SSPR, VPN, Outlook, Salesforce, JH, SFDC, Jump Host)
3. Preserve error codes verbatim (error 809, error 1603, etc.)
4. Focus on symptoms: "can't access", "won't connect", "failed", "locked", "denied", "not working"
5. Include natural synonyms where helpful: login/access, connection/connectivity, reset/unlock
6. Remove: personal pronouns (I, my, we), conversational phrases (please, urgent), person names, dates/times
7. Add context words when helpful: troubleshooting, setup, configuration, access, remote
8. Keep it 5-15 words, simple present tense
9. Use lowercase unless it's a proper product name

OUTPUT: Return ONLY the query. No explanations, no JSON.

EXAMPLES:

Input:
SUBJECT: "Can't access email"
DESCRIPTION: "User reports Outlook won't open, error says profile cannot be loaded"
CATEGORY: "Email"
Output: Outlook profile cannot be loaded troubleshooting

Input:
SUBJECT: "VPN connection failed"
DESCRIPTION: "Getting error 809 when connecting from home. Worked yesterday."
CATEGORY: "Network"
Output: VPN connection error 809 remote access troubleshooting

Input:
SUBJECT: "Software installation issue"
DESCRIPTION: "Microsoft Teams installation failing with error code 1603"
CATEGORY: "Software"
Output: Microsoft Teams installation error 1603 troubleshooting

Search Query:"""

        logger.debug("Generating Knowledge Base search query")

        response = await llm_service.invoke(
            prompt=prompt,
            max_tokens=150,
            temperature=0.1  # Low temperature for consistent extraction
        )

        # Clean response
        query = response.strip()
        # Remove any quotes if LLM added them
        query = query.strip('"').strip("'")
        # Take only first sentence if multiple returned
        if '?' in query:
            query = query.split('?')[0] + '?'

        logger.info(f"Generated Knowledge Base search query: {query}")
        return query

    except Exception as e:
        logger.error(f"Error generating Knowledgebase search query: {e}", exc_info=True)
        return _fallback_confluence_keyword_extraction(category, subject, description)



def _fallback_confluence_keyword_extraction(
    category: str,
    subject: str,
    description: str
) -> str:
    """
    Fallback keyword extraction for Confluence Memory if LLM fails

    Args:
        category: Ticket category
        subject: Ticket subject
        description: Ticket description

    Returns:
        Simple concatenated keywords
    """
    logger.warning("Using fallback  keyword extraction")

    # Combine category + subject + first 50 words of description
    parts = [category, subject]

    if description:
        words = description.split()[:50]
        parts.append(' '.join(words))

    combined = ' '.join(parts)

    # Basic cleanup
    import re
    combined = re.sub(r'[^a-zA-Z0-9\s]', ' ', combined)
    combined = ' '.join(combined.split())

    return combined[:150].lower()