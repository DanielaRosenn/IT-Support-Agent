"""
Pydantic Models for Response Evaluation

This module defines:
1. StaticEvaluationData - Extracts and structures data from GraphState for evaluation
2. ResponseEvaluation - LLM evaluation output model
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# STATIC EVALUATION DATA - Extracted from GraphState
# ============================================================================

class TicketSnapshot(BaseModel):
    """Snapshot of ticket information for evaluation"""
    ticket_id: str
    description: str
    category: Optional[str] = None
    subject: Optional[str] = None
    requester: Optional[str] = None


class KnowledgeSourceSnapshot(BaseModel):
    """Snapshot of a single knowledge source with relevance score"""
    source_type: str  # "memory", "freshservice", "context_grounding", "web_search"
    content: str
    title: Optional[str] = None
    score: Optional[float] = None  # Relevance score if available
    url: Optional[str] = None


class ResponseSnapshot(BaseModel):
    """Snapshot of the generated response"""
    response_type: str  # "self_service", "it_execution", "investigation"
    content: str
    response_length: int  # Character count
    has_numbered_steps: bool
    step_count: Optional[int] = None


class StaticEvaluationData(BaseModel):
    """
    Static data extracted from GraphState for response evaluation.
    This object is passed to the LLM along with the evaluation prompt.

    NO additional tool calls should be made - all data comes from state.
    """

    # ===== TICKET INFORMATION =====
    ticket: TicketSnapshot

    # ===== RESPONSE INFORMATION =====
    response: ResponseSnapshot

    # ===== KNOWLEDGE SOURCES USED =====
    knowledge_sources: List[KnowledgeSourceSnapshot] = Field(default_factory=list)
    knowledge_sources_count: int = 0

    # ===== KNOWLEDGE SUFFICIENCY METRICS =====
    knowledge_sufficiency_score: Optional[float] = None  # Best score from evaluate_knowledge_sufficiency
    knowledge_was_sufficient: bool = False  # If we skipped web search

    # ===== AUGMENTATION TRACKING =====
    augmentation_iterations: int = 0  # How many augmentation loops (0-2)
    augmentation_source: Optional[str] = None  # "web_topics" | "missing_info"
    web_search_used: bool = False  # If we triggered web search fallback

    # ===== IT ACTION CONTEXT =====
    was_it_action_match: bool = False
    matched_it_action_name: Optional[str] = None
    it_action_confidence: Optional[float] = None

    # ===== MEMORY CONTEXT =====
    memory_results_count: int = 0
    memory_best_score: Optional[float] = None

    # ===== METADATA =====
    evaluation_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "ticket": {
                    "ticket_id": "12345",
                    "description": "Cannot access Slack on laptop after returning from PTO",
                    "category": "Access",
                    "subject": "Slack access issue",
                    "requester": "Amelia Brown"
                },
                "response": {
                    "response_type": "self_service",
                    "content": "Hi Amelia,\n\nPlease try these steps:\n1. Connect to VPN\n2. Restart Slack\n3. Log in again",
                    "response_length": 95,
                    "has_numbered_steps": True,
                    "step_count": 3
                },
                "knowledge_sources": [
                    {
                        "source_type": "context_grounding",
                        "content": "To fix Slack connectivity...",
                        "title": "Slack VPN Requirements",
                        "score": 0.85
                    }
                ],
                "knowledge_sources_count": 1,
                "knowledge_sufficiency_score": 0.85,
                "knowledge_was_sufficient": True,
                "augmentation_iterations": 0,
                "web_search_used": False,
                "was_it_action_match": False,
                "memory_results_count": 1,
                "memory_best_score": 0.72
            }
        }


# ============================================================================
# RESPONSE EVALUATION OUTPUT - LLM generates this
# ============================================================================

class ResponseEvaluation(BaseModel):
    """
    LLM-generated evaluation of the response quality.
    This is returned by the evaluate_response node.
    """

    # ===== EVALUATION STATUS =====
    evaluated: bool = True

    # ===== RESPONSE CLASSIFICATION =====
    response_type: str  # "self_service", "it_execution", "investigation"

    # ===== CORE QUALITY METRICS (0.0-1.0) =====
    quality_score: float = Field(
        description="Overall quality of the response (0.0-1.0)",
        ge=0.0, le=1.0
    )
    completeness_score: float = Field(
        description="How complete the response is (0.0-1.0)",
        ge=0.0, le=1.0
    )
    confidence_score: float = Field(
        description="Confidence in the response based on knowledge sources (0.0-1.0)",
        ge=0.0, le=1.0
    )
    overall_score: float = Field(
        description="Weighted average of all scores (0.0-1.0)",
        ge=0.0, le=1.0
    )

    # ===== RESPONSE-TYPE-SPECIFIC METRICS (0.0-1.0) =====
    clarity_score: Optional[float] = Field(
        None,
        description="For self-service: Are instructions clear? (0.0-1.0)",
        ge=0.0, le=1.0
    )
    actionability_score: Optional[float] = Field(
        None,
        description="For IT execution: Are steps actionable? (0.0-1.0)",
        ge=0.0, le=1.0
    )
    diagnostic_depth_score: Optional[float] = Field(
        None,
        description="For investigation: Is diagnostic approach thorough? (0.0-1.0)",
        ge=0.0, le=1.0
    )

    # ===== QUALITATIVE ASSESSMENT =====
    strengths: List[str] = Field(
        default_factory=list,
        description="List of 2-4 specific strengths in the response"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="List of 1-3 specific weaknesses or areas for improvement"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="List of 0-3 elements that should be present but aren't"
    )

    # ===== KNOWLEDGE SOURCE QUALITY =====
    knowledge_assessment: str = Field(
        description="Brief assessment of knowledge source quality and relevance"
    )

    # ===== IMPROVEMENT RECOMMENDATIONS =====
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="List of 1-3 specific, actionable improvements"
    )

    # ===== CONFIDENCE LEVEL =====
    confidence_level: str = Field(
        description="Overall confidence level: 'high', 'medium', or 'low'"
    )

    # ===== EVALUATION NOTES =====
    evaluation_notes: Optional[str] = Field(
        None,
        description="Additional notes or context about the evaluation"
    )

    # ===== METADATA =====
    evaluation_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "evaluated": True,
                "response_type": "self_service",
                "quality_score": 0.85,
                "completeness_score": 0.80,
                "confidence_score": 0.85,
                "overall_score": 0.83,
                "clarity_score": 0.88,
                "strengths": [
                    "Clear step-by-step instructions",
                    "Includes context about VPN requirement",
                    "Professional and friendly tone"
                ],
                "weaknesses": [
                    "Could include estimated time to complete",
                    "Missing verification step"
                ],
                "missing_elements": [
                    "Time estimate",
                    "Verification step"
                ],
                "knowledge_assessment": "Good quality sources with high relevance (0.85 score). Context Grounding article directly addresses VPN-Slack connectivity.",
                "improvement_suggestions": [
                    "Add estimated time: '(typically takes 2-3 minutes)'",
                    "Add verification: 'Confirm you can access channels and send messages'"
                ],
                "confidence_level": "high",
                "evaluation_notes": "Strong response with minor improvements for completeness",
                "evaluation_timestamp": "2025-10-26T14:30:00Z"
            }
        }