"""
LLM Service Module
Centralized service for AWS Bedrock Claude interactions via UiPath LLM Gateway for full tracing

Includes cost tracking for token usage and LLM operations.
"""

import logging
import os
from typing import Optional, Dict, Any, List
from uipath_langchain.chat import UiPathChat
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from src.utils.exceptions import LLMError, ConfigurationError

logger = logging.getLogger(__name__)


class TokenCountingCallback(BaseCallbackHandler):
    """Callback handler to track token usage from LLM responses"""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes - extract token counts"""
        try:
            # LangChain's standard location for token counts
            if hasattr(response, 'llm_output') and response.llm_output:
                usage = response.llm_output.get('usage', {})
                if usage:
                    self.input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                    self.output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                    self.total_tokens = usage.get('total_tokens', 0) or (self.input_tokens + self.output_tokens)
                    logger.debug(f"[Callback] Captured tokens: {self.input_tokens} in + {self.output_tokens} out")
        except Exception as e:
            logger.debug(f"[Callback] Failed to extract tokens: {e}")

# Model pricing (USD per 1M tokens) - Update these if pricing changes
MODEL_PRICING = {
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {
        "input": 3.00,   # $3 per 1M input tokens
        "output": 15.00  # $15 per 1M output tokens
    },
    "us.anthropic.claude-sonnet-3-5-20241022-v2:0": {
        "input": 3.00,
        "output": 15.00
    },
    # Default fallback pricing
    "default": {
        "input": 3.00,
        "output": 15.00
    }
}

# Keep ChatBedrockConverse import for debugging/fallback
try:
    import boto3
    from langchain_aws import ChatBedrockConverse
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    logger.warning("ChatBedrockConverse not available - using UiPathChat only")


class LLMService:
    """Async service for AWS Bedrock via UiPath LLM Gateway - enables Orchestrator tracing

    Features:
    - Token usage tracking
    - Cost calculation per operation
    - Aggregated cost reporting
    """

    def __init__(self, use_direct_bedrock: bool = False):
        """Initialize LLM service with UiPath tracing support

        Args:
            use_direct_bedrock: If True, use ChatBedrockConverse directly (no Orchestrator traces)
                               If False, use UiPathChat via LLM Gateway (full Orchestrator traces)
        """
        self.model_id = os.getenv('BEDROCK_MODEL_ID', 'us.anthropic.claude-sonnet-4-20250514-v1:0')
        self.region_name = os.getenv('AWS_REGION', 'us-east-1')
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.use_direct_bedrock = use_direct_bedrock

        # Cost tracking
        self.token_usage: List[Dict[str, Any]] = []
        self._reset_tracking()

        # Validate required credentials
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ConfigurationError(
                "AWS credentials not configured",
                details={
                    "missing": [
                        k for k, v in {
                            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
                            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key
                        }.items() if not v
                    ]
                }
            )

        # Always use ChatBedrockConverse for now (handles Bedrock directly with tracing support)
        logger.info("Using ChatBedrockConverse for AWS Bedrock with LangChain tracing")
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )

        self.llm = ChatBedrockConverse(
            model=self.model_id,
            client=bedrock_client,
            temperature=0.3,
            max_tokens=4000
        )
        logger.info(f"ChatBedrockConverse initialized: {self.model_id}")

    async def invoke(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        system_message: Optional[str] = None,
        operation_name: Optional[str] = None
    ) -> str:
        """
        Invoke Claude model with a prompt via LangChain (async) - enables UiPath tracing

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response (overrides default)
            temperature: Temperature (0-1, overrides default)
            system_message: Optional system message to set context/behavior
            operation_name: Optional name for cost tracking (e.g., "check_it_actions")

        Returns:
            Model response text

        Raises:
            Exception: If invocation fails
        """
        messages = []

        # Add system message if provided
        if system_message:
            messages.append(SystemMessage(content=system_message))

        # Add user prompt
        messages.append(HumanMessage(content=prompt))

        logger.debug(f"Invoking Bedrock via LangChain: {self.model_id}")

        # Create callback to capture token usage
        token_callback = TokenCountingCallback()

        try:
            # Invoke via LangChain with callback (enables automatic UiPath tracing)
            response = await self.llm.ainvoke(messages, config={"callbacks": [token_callback]})
            text = response.content

            # Handle multiple response formats from Bedrock
            if isinstance(text, list):
                # List of content blocks (e.g., [{'type': 'text', 'text': '...', 'index': 0}])
                if len(text) > 0:
                    # Extract text from first content block if it's a dict
                    first_item = text[0]
                    if isinstance(first_item, dict) and 'text' in first_item:
                        text = first_item['text']
                    else:
                        text = str(first_item)
                else:
                    text = ""
            elif isinstance(text, dict):
                # Single content block as dict: {'type': 'text', 'text': '...', 'index': 0}
                if 'text' in text:
                    text = text['text']
                else:
                    text = str(text)
            # else: already a string, use as-is

            logger.debug(f"Bedrock response received ({len(text)} chars)")

            # Track token usage and costs (pass callback for token data)
            self._track_usage(response, operation_name or "unknown", token_callback, prompt, text)

            return text.strip()

        except Exception as e:
            # Wrap any errors for consistent error handling
            logger.error(f"Failed to invoke Bedrock LLM: {e}", exc_info=True)
            raise LLMError(
                f"Failed to invoke Bedrock LLM: {str(e)}",
                details={
                    "operation": "invoke",
                    "prompt_length": len(prompt),
                    "model_id": self.model_id
                }
            )

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.llm is not None

    def _track_usage(
        self,
        response: Any,
        operation_name: str,
        callback: TokenCountingCallback,
        prompt: str,
        response_text: str
    ) -> None:
        """
        Track token usage and cost from LLM response.
        Tries multiple methods in order of reliability:
        1. Callback data (from LangChain callbacks)
        2. Response metadata fields
        3. Estimation based on text length

        Args:
            response: LangChain response object
            operation_name: Name of the operation for tracking
            callback: Token counting callback with captured usage
            prompt: Input prompt text (for estimation fallback)
            response_text: Output response text (for estimation fallback)
        """
        try:
            input_tokens = 0
            output_tokens = 0
            method = "unknown"

            # METHOD 1: Try callback data first (most reliable)
            if callback.total_tokens > 0:
                input_tokens = callback.input_tokens
                output_tokens = callback.output_tokens
                method = "callback"
                logger.debug(f"[Cost Tracking] Using callback: input={input_tokens}, output={output_tokens}")

            # METHOD 2: Try response.usage_metadata (LangChain standard)
            elif hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_meta = response.usage_metadata
                # Handle both dict and object
                if isinstance(usage_meta, dict):
                    input_tokens = usage_meta.get('input_tokens', 0)
                    output_tokens = usage_meta.get('output_tokens', 0)
                else:
                    input_tokens = getattr(usage_meta, 'input_tokens', 0)
                    output_tokens = getattr(usage_meta, 'output_tokens', 0)
                method = "usage_metadata"
                logger.debug(f"[Cost Tracking] Using usage_metadata: input={input_tokens}, output={output_tokens}")

            # METHOD 3: Try response_metadata (Bedrock specific)
            elif hasattr(response, 'response_metadata'):
                metadata = response.response_metadata

                # Try usage field in metadata
                if isinstance(metadata, dict) and 'usage' in metadata:
                    usage = metadata['usage']
                    input_tokens = usage.get('inputTokens', 0) or usage.get('input_tokens', 0)
                    output_tokens = usage.get('outputTokens', 0) or usage.get('output_tokens', 0)
                    method = "response_metadata.usage"
                    logger.debug(f"[Cost Tracking] Using response_metadata.usage: input={input_tokens}, output={output_tokens}")

            # METHOD 4: Estimate based on text length (last resort)
            if input_tokens == 0 and output_tokens == 0:
                input_tokens = self._estimate_tokens(prompt)
                output_tokens = self._estimate_tokens(response_text)
                method = "estimation"
                logger.warning(
                    f"[Cost Tracking] No metadata available for {operation_name}. "
                    f"Using estimation: input={input_tokens}, output={output_tokens}"
                )

            total_tokens = input_tokens + output_tokens

            # Calculate cost
            cost_usd = self._calculate_cost(input_tokens, output_tokens)

            # Store usage record
            usage_record = {
                "operation": operation_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "model": self.model_id,
                "tracking_method": method
            }
            self.token_usage.append(usage_record)

            logger.info(
                f"[Cost Tracking] {operation_name}: "
                f"{input_tokens} in + {output_tokens} out = {total_tokens} total tokens "
                f"(${cost_usd:.4f}) [method: {method}]"
            )

        except Exception as e:
            logger.warning(f"[Cost Tracking] Failed to track token usage: {e}")
            # Don't fail the request if tracking fails

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text length.
        Uses rough approximation: 1 token ≈ 4 characters for English text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Claude models: ~4 chars per token on average
        # Add 10% buffer for safety
        estimated = int(len(text) / 4 * 1.1)
        return max(1, estimated)  # Minimum 1 token

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost in USD based on token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        # Get pricing for current model (or use default)
        pricing = MODEL_PRICING.get(self.model_id, MODEL_PRICING["default"])

        # Convert pricing from per-1M-tokens to per-token
        input_price_per_token = pricing["input"] / 1_000_000
        output_price_per_token = pricing["output"] / 1_000_000

        # Calculate costs
        input_cost = input_tokens * input_price_per_token
        output_cost = output_tokens * output_price_per_token

        return input_cost + output_cost

    def get_total_costs(self) -> Dict[str, Any]:
        """
        Get aggregated cost data for all LLM operations.

        Returns:
            Dict with total costs, token counts, and breakdown by operation
        """
        if not self.token_usage:
            return {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
                "model": self.model_id,
                "llm_calls_count": 0,
                "breakdown": []
            }

        # Aggregate totals
        total_input = sum(u["input_tokens"] for u in self.token_usage)
        total_output = sum(u["output_tokens"] for u in self.token_usage)
        total_tokens = sum(u["total_tokens"] for u in self.token_usage)
        total_cost = sum(u["cost_usd"] for u in self.token_usage)

        return {
            "total_tokens": total_tokens,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost_usd": round(total_cost, 4),
            "model": self.model_id,
            "llm_calls_count": len(self.token_usage),
            "average_cost_per_call": round(total_cost / len(self.token_usage), 4) if self.token_usage else 0.0,
            "breakdown": self.token_usage
        }

    def _reset_tracking(self) -> None:
        """Reset cost tracking (for testing or new sessions)"""
        self.token_usage = []

    def get_cost_summary(self) -> str:
        """
        Get a human-readable cost summary.

        Returns:
            Formatted string with cost breakdown
        """
        costs = self.get_total_costs()

        if costs["llm_calls_count"] == 0:
            return "No LLM calls tracked"

        summary = f"""
LLM Cost Summary:
  Model: {costs['model']}
  Total Calls: {costs['llm_calls_count']}
  Total Tokens: {costs['total_tokens']:,} ({costs['input_tokens']:,} in + {costs['output_tokens']:,} out)
  Estimated Cost: ${costs['estimated_cost_usd']:.4f}
  Average per Call: ${costs['average_cost_per_call']:.4f}
"""

        if costs['breakdown']:
            summary += "\n  Operations Breakdown:"
            for record in costs['breakdown']:
                summary += f"\n    - {record['operation']}: {record['total_tokens']} tokens (${record['cost_usd']:.4f})"

        return summary.strip()


# Global instance
_llm_service = None


def get_llm_service() -> LLMService:
    """Get global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service