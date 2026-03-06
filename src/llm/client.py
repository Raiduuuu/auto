"""
LLM Client - Integration with language models for agent reasoning

Updated March 2026:
- Uses anthropic.AsyncAnthropic for native async support
- Supports Claude Opus 4.6, Sonnet 4.6, Haiku 4.5
- Tool use is GA (no beta header needed)
- Structured outputs support
"""
import re
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

    Latest Claude models (March 2026):
    - claude-opus-4-6: Most capable, best for complex reasoning
    - claude-sonnet-4-6: Balanced performance/cost
    - claude-haiku-4-5: Fastest, lowest cost
    """

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        temperature: float = 0.7
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize async client
        if provider == "anthropic" and anthropic:
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
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
        """Send analysis request to LLM."""
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
        """Make request to Anthropic API using native async client."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
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

    async def analyze_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: str,
        media_type: str = "image/png",
        temperature: Optional[float] = None
    ) -> str:
        """Send analysis request with image (vision)."""
        if not self.client:
            return "LLM client not available"

        temp = temperature if temperature is not None else self.temperature

        try:
            if self.provider == "anthropic":
                return await self._anthropic_vision_request(
                    system_prompt, user_prompt, image_data, media_type, temp
                )
            else:
                return "Vision not supported for this provider"
        except Exception as e:
            logger.error(f"Vision request failed: {e}")
            return f"Error: {str(e)}"

    async def _anthropic_vision_request(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data: str,
        media_type: str,
        temperature: float
    ) -> str:
        """Make vision request to Anthropic API."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )

        return response.content[0].text

    async def analyze_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Send request with tool definitions for structured function calling.
        Tool use is GA since late 2025 - no beta headers needed.
        """
        if not self.client or self.provider != "anthropic":
            return {"error": "Tool use requires Anthropic client"}

        temp = temperature if temperature is not None else self.temperature

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temp,
                system=system_prompt,
                tools=tools,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Parse tool use blocks
            result = {
                "stop_reason": response.stop_reason,
                "content": [],
                "tool_calls": []
            }

            for block in response.content:
                if block.type == "text":
                    result["content"].append(block.text)
                elif block.type == "tool_use":
                    result["tool_calls"].append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })

            return result

        except Exception as e:
            logger.error(f"Tool use request failed: {e}")
            return {"error": str(e)}

    async def generate_trading_decision(
        self,
        context: Dict[str, Any],
        system_prompt: str
    ) -> Dict[str, Any]:
        """Generate a structured trading decision."""
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

        return {
            "raw_response": response,
            "parsed": self._parse_decision(response)
        }

    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured decision."""
        response_lower = response.lower()

        if "buy" in response_lower:
            action = "BUY"
        elif "sell" in response_lower:
            action = "SELL"
        else:
            action = "HOLD"

        confidence_match = re.search(r'(\d+)%', response)
        confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.5

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": response[:500]
        }

    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self.client is not None
