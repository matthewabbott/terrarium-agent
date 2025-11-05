#!/usr/bin/env python3
"""Main entry point for Terrarium Agent."""

import asyncio
import sys
from pathlib import Path

from llm.vllm_client import VLLMClient
from agent.runtime import AgentRuntime
from agent.context import ContextManager
from tools.irc import IRCTool


async def main():
    """Initialize and run agent."""
    print("="*60)
    print("Terrarium Agent")
    print("="*60)

    # Initialize vLLM client
    print("\nInitializing vLLM client...")
    llm = VLLMClient(
        base_url="http://localhost:8000",
        model="glm-air-4.5",
        temperature=0.7,
        max_tokens=2048
    )

    # Check vLLM server health
    print("Checking vLLM server...")
    is_healthy = await llm.health_check()
    if not is_healthy:
        print("ERROR: vLLM server not responding at http://localhost:8000")
        print("Make sure vLLM is running:")
        print("  vllm serve glm-air-4.5 --quantization awq --dtype half")
        return

    # Get available models
    models = await llm.get_models()
    print(f"Available models: {', '.join(models)}")

    # Initialize context manager
    print("\nInitializing context manager...")
    context_manager = ContextManager(contexts_dir=Path("config/contexts"))

    # Create agent runtime
    print("\nCreating agent runtime...")
    agent = AgentRuntime(llm_client=llm, context_manager=context_manager)

    # Register tools
    print("\nRegistering tools...")
    agent.register_tool(IRCTool())

    # Initialize agent
    print("\nInitializing agent runtime...")
    await agent.initialize()

    print("\n" + "="*60)
    print("Agent ready!")
    print("="*60)

    # Interactive loop
    print("\nEntering interactive mode. Type 'quit' to exit.")
    print("Commands:")
    print("  /context <name>  - Switch context")
    print("  /list            - List available contexts")
    print("  /tools           - List registered tools")
    print()

    try:
        while True:
            # Get user input
            prompt = input("You: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            # Handle commands
            if prompt.startswith('/'):
                parts = prompt.split(None, 1)
                command = parts[0][1:]

                if command == 'context':
                    if len(parts) < 2:
                        print("Usage: /context <name>")
                        continue
                    context_name = parts[1]
                    if agent.context_manager.switch_context(context_name):
                        print(f"Switched to context: {context_name}")
                    else:
                        print(f"Context not found: {context_name}")

                elif command == 'list':
                    contexts = agent.context_manager.list_contexts()
                    current = agent.context_manager.current_context
                    print("Available contexts:")
                    for ctx in contexts:
                        marker = " *" if current and ctx == current.name else ""
                        print(f"  - {ctx}{marker}")

                elif command == 'tools':
                    print("Registered tools:")
                    for tool_name, tool in agent.tools.items():
                        status = "enabled" if tool.enabled else "disabled"
                        print(f"  - {tool_name} ({status})")
                        print(f"    {tool.description}")

                else:
                    print(f"Unknown command: {command}")

                continue

            # Process prompt with agent
            print("Agent: ", end="", flush=True)
            response = await agent.process(prompt)
            print(response)
            print()

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        # Cleanup
        await agent.shutdown()
        print("Agent stopped.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        sys.exit(0)
