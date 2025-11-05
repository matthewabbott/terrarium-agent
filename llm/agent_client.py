"""Agent client for external invocation (e.g., from terrarium-irc).

This module provides a simple interface for invoking terrarium-agent
from external tools like the IRC bot, without running the full interactive
agent loop.
"""

from typing import Optional
from pathlib import Path

from .vllm_client import VLLMClient
from agent.runtime import AgentRuntime
from agent.context import ContextManager
from tools.irc import IRCTool
from tools.harness import HarnessAdapter
from tools.harness_examples import NumberGuessHarness, TextAdventureHarness


class AgentClient:
    """
    Client interface for invoking terrarium-agent from external tools.

    Designed for on-demand agent invocation with minimal overhead.
    Handles agent lifecycle: initialize → process → (optional) shutdown.

    Usage:
        # One-off query
        client = AgentClient()
        response = await client.generate("What is 2+2?")
        await client.shutdown()

        # Or use as context manager
        async with AgentClient() as client:
            response = await client.generate("Hello!")

        # With IRC context
        irc_history = "[12:00] <alice> What's the weather?\n[12:01] <bob> Sunny!"
        response = await client.generate(
            "What did alice ask?",
            context=irc_history
        )
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        vllm_model: str = "glm-air-4.5",
        contexts_dir: Optional[Path] = None,
        register_tools: bool = True
    ):
        """
        Initialize agent client.

        Args:
            vllm_url: vLLM server URL
            vllm_model: Model name (as configured in vLLM)
            contexts_dir: Path to context definitions (auto-detected if None)
            register_tools: Whether to register default tools (IRC, harnesses)
        """
        self.vllm_url = vllm_url
        self.vllm_model = vllm_model

        # Auto-detect contexts directory
        if contexts_dir is None:
            contexts_dir = Path(__file__).parent.parent / "config" / "contexts"
        self.contexts_dir = contexts_dir

        self.register_tools = register_tools

        # Runtime is created on-demand
        self._runtime: Optional[AgentRuntime] = None
        self._initialized = False

    async def initialize(self):
        """Initialize agent runtime (lazy initialization)."""
        if self._initialized:
            return

        print("[AgentClient] Initializing...")

        # Create vLLM client
        llm = VLLMClient(
            base_url=self.vllm_url,
            model=self.vllm_model,
            temperature=0.7,
            max_tokens=2048
        )

        # Create context manager
        context_manager = ContextManager(contexts_dir=self.contexts_dir)

        # Create runtime
        self._runtime = AgentRuntime(
            llm_client=llm,
            context_manager=context_manager
        )

        # Register tools if requested
        if self.register_tools:
            # IRC tool
            self._runtime.register_tool(IRCTool())

            # Harnesses
            self._runtime.register_tool(HarnessAdapter(NumberGuessHarness()))
            self._runtime.register_tool(HarnessAdapter(TextAdventureHarness()))

        # Initialize runtime
        await self._runtime.initialize()

        self._initialized = True
        print("[AgentClient] Ready.")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        context_name: str = "irc"
    ) -> str:
        """
        Generate response using agent.

        Args:
            prompt: User prompt/question
            system_prompt: Override system prompt (optional, uses context otherwise)
            context: Additional context (e.g., IRC chat history)
            context_name: Context to load (default: "irc")

        Returns:
            Generated response

        Raises:
            Exception: If vLLM server is not responding or generation fails
        """
        # Ensure initialized
        await self.initialize()

        # Build full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\nQuestion: {prompt}"

        # Process with agent
        try:
            response = await self._runtime.process(
                prompt=full_prompt,
                context_name=context_name
            )
            return response

        except Exception as e:
            raise Exception(f"Agent generation failed: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if vLLM server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            # Create temporary vLLM client for health check
            llm = VLLMClient(base_url=self.vllm_url)
            return await llm.health_check()
        except Exception:
            return False

    async def shutdown(self):
        """Shutdown agent runtime and cleanup resources."""
        if self._runtime:
            print("[AgentClient] Shutting down...")
            await self._runtime.shutdown()
            self._runtime = None
            self._initialized = False
            print("[AgentClient] Shutdown complete.")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"<AgentClient(vllm={self.vllm_url}, status={status})>"


# Example usage
async def example_usage():
    """Example of how to use AgentClient."""
    import asyncio

    # Create client
    client = AgentClient()

    # Check if vLLM is available
    if not await client.health_check():
        print("vLLM server not available at http://localhost:8000")
        print("Start it with: cd terrarium-agent && ./start_vllm.sh")
        return

    # Generate response
    print("Asking agent...")
    response = await client.generate(
        prompt="What is the capital of France?",
        context_name="irc"
    )
    print(f"Response: {response}")

    # With IRC context
    irc_context = """
[10:23] <alice> What's the weather like today?
[10:24] <bob> It's sunny and warm!
[10:25] <alice> Perfect for a picnic.
"""

    print("\nAsking agent with IRC context...")
    response = await client.generate(
        prompt="What did alice and bob discuss?",
        context=irc_context
    )
    print(f"Response: {response}")

    # Cleanup
    await client.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
