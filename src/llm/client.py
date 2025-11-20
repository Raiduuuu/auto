"""
LLM Client - Integration with language models for agent reasoning
"""
import asyncio
from typing import Any, Dict, Optional
from loguru import logger

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None


class LLMClient:
    """
    Client for LLM API interactions.
    Supports Claude (Anthropic) and GPT (OpenAI).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.7
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize client
        if provider == "anthropic" and anthropic:
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai" and openai:
            self.client = openai.AsyncOpenAI(api_key=api_key)
        else:
            self.client = None
            logger.warning(f"LLM client not initialized: {provider}")

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Send analysis request to LLM.
        """
        if not self.client:
            return "LLM client not available"

        temp = temperature if temperature is not None else self.temperature

        try:
            if self.provider == "anthropic":
                return await self._anthropic_request(system_prompt, user_prompt, temp)
            elif self.provider == "openai":
                return await self._openai_request(system_prompt, user_prompt, temp)
            else:
                return "Unsupported provider"
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return f"Error: {str(e)}"

    async def _anthropic_request(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> str:
        """Make request to Anthropic API."""
        # Run sync client in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        )

        return response.content[0].text

    async def _openai_request(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> str:
        """Make request to OpenAI API."""
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content

    async def generate_trading_decision(
        self,
        context: Dict[str, Any],
        system_prompt: str
    ) -> Dict[str, Any]:
        """
        Generate a structured trading decision.
        """
        user_prompt = f"""Based on the following market context, provide a trading decision:

Context:
{context}

Respond with:
1. Action: BUY, SELL, or HOLD
2. Confidence: 0-100%
3. Entry price (if trading)
4. Stop loss
5. Take profit
6. Brief reasoning (2-3 sentences)"""

        response = await self.analyze(system_prompt, user_prompt)

        # Parse response (simple extraction)
        return {
            "raw_response": response,
            "parsed": self._parse_decision(response)
        }

    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured decision."""
        response_lower = response.lower()

        # Extract action
        if "buy" in response_lower:
            action = "BUY"
        elif "sell" in response_lower:
            action = "SELL"
        else:
            action = "HOLD"

        # Extract confidence (look for percentage)
        import re
        confidence_match = re.search(r'(\d+)%', response)
        confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": response[:500]  # Truncate
        }

    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self.client is not None
