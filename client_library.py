"""
Terrarium Agent HTTP Client Library

Simple client for calling the Terrarium Agent HTTP API server.
External applications (terrarium-irc, web apps, games) use this to get LLM responses.

This is for the HTTP server-based approach (server.py running on port 8080).
For the library-based approach, see llm/agent_client.py instead.

Usage:
    from client_library import AgentClient

    client = AgentClient("http://localhost:8080")

    # Simple generation
    response = client.generate("What is Python?")

    # Chat with history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    response = client.chat(messages)
"""

import time
import json
from typing import List, Dict, Optional
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError


class AgentClientError(Exception):
    """Base exception for agent client errors."""
    pass


class AgentServerError(AgentClientError):
    """Server-side error (5xx)."""
    pass


class AgentRequestError(AgentClientError):
    """Client-side error (4xx)."""
    pass


class AgentConnectionError(AgentClientError):
    """Connection error (server unavailable)."""
    pass


class AgentClient:
    """
    Client for Terrarium Agent HTTP API.

    Provides simple interface for generating responses from the agent server.
    Handles retries, timeouts, and error formatting.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize agent client.

        Args:
            base_url: Agent server URL (default: http://localhost:8080)
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum retry attempts for failed requests (default: 3)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: Optional[str] = None
    ) -> str:
        """
        Generate response with full conversation history.

        Args:
            messages: Conversation history (list of {"role": "...", "content": "..."})
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 2048)
            model: Model name (auto-detected if omitted)

        Returns:
            Assistant's response text

        Raises:
            AgentRequestError: Invalid request (4xx)
            AgentServerError: Server error (5xx)
            AgentConnectionError: Cannot connect to server
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if model:
            payload["model"] = model

        response_data = self._request_with_retry(
            "POST",
            "/v1/chat/completions",
            json=payload
        )

        # Extract response text from OpenAI-compatible format
        try:
            return response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise AgentClientError(f"Invalid response format: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Simple single-turn generation.

        Args:
            prompt: User's message
            system_prompt: Optional system prompt
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 512)

        Returns:
            Assistant's response text

        Raises:
            AgentRequestError: Invalid request (4xx)
            AgentServerError: Server error (5xx)
            AgentConnectionError: Cannot connect to server
        """
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        response_data = self._request_with_retry(
            "POST",
            "/v1/generate",
            json=payload
        )

        return response_data["response"]

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: Optional[str] = None
    ):
        """
        Stream chat completion chunks (Server-Sent Events).

        Yields response text incrementally as it's generated.

        Args:
            messages: Conversation history (list of {"role": "...", "content": "..."})
            temperature: Sampling temperature 0.0-2.0 (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 2048)
            model: Model name (auto-detected if omitted)

        Yields:
            str: Text chunks as they're generated

        Raises:
            AgentRequestError: Invalid request (4xx)
            AgentServerError: Server error (5xx)
            AgentConnectionError: Cannot connect to server
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True  # Enable streaming
        }

        if model:
            payload["model"] = model

        url = f"{self.base_url}/v1/chat/completions"

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=True  # Enable streaming in requests
            )

            # Handle errors
            if response.status_code >= 400:
                error_data = self._extract_error(response)
                if response.status_code >= 500:
                    raise AgentServerError(f"Server error: {error_data}")
                else:
                    raise AgentRequestError(f"Request error: {error_data}")

            # Parse SSE stream
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.strip():
                    continue

                # SSE format: "data: {json}"
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix

                    # Check for stream end
                    if data == '[DONE]':
                        break

                    # Parse chunk
                    try:
                        chunk = json.loads(data)

                        # Extract text from chunk
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]

                            # Streaming chunks have "delta" instead of "message"
                            if "delta" in choice:
                                content = choice["delta"].get("content", "")
                                if content:
                                    yield content

                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        continue

        except ConnectionError:
            raise AgentConnectionError(f"Cannot connect to agent server at {self.base_url}")
        except Timeout:
            raise AgentConnectionError(f"Request timed out after {self.timeout}s")
        except RequestException as e:
            raise AgentClientError(f"Stream request failed: {e}")

    def health_check(self) -> Dict[str, any]:
        """
        Check server health.

        Returns:
            Health status dictionary with keys:
                - status: "healthy" or "degraded"
                - vllm_status: "ready", "unavailable", etc.
                - model: Model name
                - queue_length: Number of queued requests

        Raises:
            AgentConnectionError: Cannot connect to server
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except ConnectionError:
            raise AgentConnectionError(f"Cannot connect to {self.base_url}")
        except Timeout:
            raise AgentConnectionError(f"Health check timed out after {self.timeout}s")
        except RequestException as e:
            raise AgentClientError(f"Health check failed: {e}")

    def get_models(self) -> List[str]:
        """
        Get list of available models.

        Returns:
            List of model names

        Raises:
            AgentConnectionError: Cannot connect to server
        """
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data["data"]]
        except ConnectionError:
            raise AgentConnectionError(f"Cannot connect to {self.base_url}")
        except Timeout:
            raise AgentConnectionError(f"Request timed out after {self.timeout}s")
        except RequestException as e:
            raise AgentClientError(f"Failed to get models: {e}")

    def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict:
        """
        Make HTTP request with exponential backoff retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/v1/chat/completions")
            **kwargs: Additional arguments for requests (json, params, etc.)

        Returns:
            Response JSON data

        Raises:
            AgentRequestError: Client error (4xx)
            AgentServerError: Server error (5xx)
            AgentConnectionError: Connection failed
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )

                # Handle errors
                if response.status_code >= 500:
                    # Server error - retry with backoff
                    if attempt < self.max_retries - 1:
                        backoff = 2 ** attempt
                        time.sleep(backoff)
                        continue
                    else:
                        # Max retries exceeded
                        error_data = self._extract_error(response)
                        raise AgentServerError(
                            f"Server error after {self.max_retries} attempts: {error_data}"
                        )

                elif response.status_code >= 400:
                    # Client error - don't retry
                    error_data = self._extract_error(response)
                    raise AgentRequestError(f"Request error: {error_data}")

                # Success
                response.raise_for_status()
                return response.json()

            except Timeout:
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
                    continue
                else:
                    raise AgentConnectionError(
                        f"Request timed out after {self.max_retries} attempts"
                    )

            except ConnectionError:
                raise AgentConnectionError(
                    f"Cannot connect to agent server at {self.base_url}"
                )

            except RequestException as e:
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
                    continue
                else:
                    raise AgentClientError(f"Request failed: {e}")

        # Should never reach here
        raise AgentClientError("Unexpected error in request handling")

    @staticmethod
    def _extract_error(response: requests.Response) -> str:
        """Extract error message from response."""
        try:
            data = response.json()
            if "error" in data:
                return data["error"].get("message", str(data))
            return str(data)
        except:
            return response.text or f"HTTP {response.status_code}"


# ============================================================================
# Conversation History Helper
# ============================================================================

class ConversationContext:
    """
    Helper for managing conversation history.

    Clients are responsible for storing their own conversation contexts.
    This class provides utilities to make that easier.
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        """
        Initialize conversation context.

        Args:
            system_prompt: System prompt for the conversation
        """
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        """Add user message to history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add assistant message to history."""
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get full message history including system prompt.

        Returns:
            List of messages ready to send to agent
        """
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]

    def clear(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = []

    def get_last_n(self, n: int) -> List[Dict[str, str]]:
        """
        Get last n messages (plus system prompt).

        Useful for keeping context window manageable.

        Args:
            n: Number of recent messages to include

        Returns:
            System prompt + last n messages
        """
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages[-n:]
        ]


# ============================================================================
# Example Usage
# ============================================================================

def example_simple_generation():
    """Example: Simple single-turn generation."""
    client = AgentClient("http://localhost:8080")

    response = client.generate(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful assistant."
    )

    print(f"Response: {response}")


def example_chat_with_history():
    """Example: Multi-turn conversation with history."""
    client = AgentClient("http://localhost:8080")
    context = ConversationContext("You are a helpful coding assistant.")

    # First turn
    context.add_user_message("What is a Python decorator?")
    response = client.chat(context.get_messages())
    context.add_assistant_message(response)
    print(f"Assistant: {response}\n")

    # Second turn (with context)
    context.add_user_message("Can you show me an example?")
    response = client.chat(context.get_messages())
    context.add_assistant_message(response)
    print(f"Assistant: {response}\n")


def example_streaming():
    """Example: Streaming chat with live typing effect."""
    client = AgentClient("http://localhost:8080")
    context = ConversationContext("You are a helpful assistant.")

    # Ask a question
    context.add_user_message("Explain how HTTP streaming works in 2-3 sentences.")

    print("Assistant: ", end="", flush=True)

    # Stream response (yields text chunks)
    full_response = ""
    try:
        for chunk in client.chat_stream(context.get_messages()):
            print(chunk, end="", flush=True)
            full_response += chunk

        print("\n")  # New line after streaming completes

        # Add to context for next turn
        context.add_assistant_message(full_response)

    except AgentClientError as e:
        print(f"\nError: {e}")


def example_irc_bot():
    """Example: IRC bot with per-channel contexts."""
    client = AgentClient("http://localhost:8080")

    # Store conversation context per channel
    channel_contexts = {}

    def on_message(channel: str, user: str, message: str):
        """Handle IRC message."""
        # Get or create context for this channel
        if channel not in channel_contexts:
            channel_contexts[channel] = ConversationContext(
                "You are a helpful IRC bot. Be concise."
            )

        context = channel_contexts[channel]

        # Add user message
        context.add_user_message(f"{user}: {message}")

        # Get response
        try:
            response = client.chat(
                messages=context.get_messages(),
                temperature=0.8,
                max_tokens=512
            )

            # Add to history
            context.add_assistant_message(response)

            # Send to IRC
            print(f"[{channel}] {response}")
            return response

        except AgentConnectionError:
            print("Agent server unavailable")
            return None
        except AgentClientError as e:
            print(f"Agent error: {e}")
            return None

    # Simulate IRC messages
    on_message("#python", "alice", "What is a decorator?")
    on_message("#python", "bob", "Can you explain async/await?")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example_name = sys.argv[1]

        if example_name == "simple":
            print("=" * 60)
            print("Example: Simple Generation")
            print("=" * 60)
            example_simple_generation()

        elif example_name == "chat":
            print("=" * 60)
            print("Example: Chat with History")
            print("=" * 60)
            example_chat_with_history()

        elif example_name == "stream":
            print("=" * 60)
            print("Example: Streaming Chat")
            print("=" * 60)
            example_streaming()

        elif example_name == "irc":
            print("=" * 60)
            print("Example: IRC Bot")
            print("=" * 60)
            example_irc_bot()

        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: simple, chat, stream, irc")

    else:
        # Run all examples
        print("=" * 60)
        print("Example 1: Simple Generation")
        print("=" * 60)
        try:
            example_simple_generation()
        except AgentConnectionError as e:
            print(f"ERROR: {e}")
            print("\nMake sure the agent server is running:")
            print("  python server.py")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Example 2: Chat with History")
        print("=" * 60)
        example_chat_with_history()

        print("\n" + "=" * 60)
        print("Example 3: IRC Bot Simulation")
        print("=" * 60)
        example_irc_bot()
