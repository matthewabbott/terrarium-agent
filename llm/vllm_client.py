"""vLLM client for GLM-Air-4.5 integration."""

import aiohttp
from typing import Optional, List, Dict, Any
import json


class VLLMClient:
    """
    Client for vLLM server running GLM-Air-4.5 (4-bit quantized).

    Uses OpenAI-compatible API endpoint that vLLM provides.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "glm-air-4.5",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120
    ):
        """
        Initialize vLLM client.

        Args:
            base_url: vLLM server URL
            model: Model name (as configured in vLLM)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        """Initialize HTTP session with improved timeout configuration."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=600,      # 10 minutes max for long responses
                    connect=10,     # 10 seconds to establish connection
                    sock_read=120   # 2 minutes between chunks (for streaming)
                )
            )

    async def shutdown(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop: List of stop sequences

        Returns:
            Generated text

        Raises:
            Exception: If generation fails
        """
        if not self.session:
            await self.initialize()

        # Build messages in chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        if stop:
            payload["stop"] = stop

        # Call vLLM API (OpenAI-compatible endpoint)
        url = f"{self.base_url}/v1/chat/completions"

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM API error ({response.status}): {error_text}")

            data = await response.json()

            # Extract generated text
            if "choices" not in data or len(data["choices"]) == 0:
                raise Exception(f"No response from vLLM: {data}")

            return data["choices"][0]["message"]["content"]

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate response with full conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     (roles: 'system', 'user', 'assistant')
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop: List of stop sequences

        Returns:
            Generated text

        Raises:
            Exception: If generation fails
        """
        if not self.session:
            await self.initialize()

        # Prepare request
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        if stop:
            payload["stop"] = stop

        # Call vLLM API (OpenAI-compatible endpoint)
        url = f"{self.base_url}/v1/chat/completions"

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM API error ({response.status}): {error_text}")

            data = await response.json()

            # Extract generated text
            if "choices" not in data or len(data["choices"]) == 0:
                raise Exception(f"No response from vLLM: {data}")

            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ):
        """
        Generate streaming response with full conversation history.

        Yields Server-Sent Events (SSE) chunks as they're generated.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop: List of stop sequences

        Yields:
            Dict chunks in OpenAI streaming format

        Raises:
            Exception: If generation fails
        """
        if not self.session:
            await self.initialize()

        # Prepare request with streaming enabled
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": True  # Enable streaming
        }

        if stop:
            payload["stop"] = stop

        url = f"{self.base_url}/v1/chat/completions"

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM API error ({response.status}): {error_text}")

            # Parse SSE stream
            async for line in response.content:
                line = line.decode('utf-8').strip()

                # SSE format: "data: {json}\n"
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix

                    # Check for stream end
                    if data == '[DONE]':
                        break

                    # Parse JSON chunk
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        continue

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate with tool calling capability.

        Args:
            prompt: User prompt
            tools: List of tool definitions (OpenAI function calling format)
            system_prompt: Optional system prompt
            temperature: Override default temperature

        Returns:
            Dict with 'text' and optionally 'tool_calls'
        """
        if not self.session:
            await self.initialize()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare request with tools
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
        }

        url = f"{self.base_url}/v1/chat/completions"

        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM API error ({response.status}): {error_text}")

            data = await response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise Exception(f"No response from vLLM: {data}")

            choice = data["choices"][0]
            message = choice["message"]

            result = {
                "text": message.get("content", ""),
                "tool_calls": message.get("tool_calls", [])
            }

            return result

    async def health_check(self) -> bool:
        """
        Check if vLLM server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        if not self.session:
            await self.initialize()

        try:
            url = f"{self.base_url}/health"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    async def get_models(self) -> List[str]:
        """
        Get list of available models.

        Returns:
            List of model names
        """
        if not self.session:
            await self.initialize()

        url = f"{self.base_url}/v1/models"
        async with self.session.get(url) as response:
            if response.status != 200:
                return []

            data = await response.json()
            return [model["id"] for model in data.get("data", [])]
