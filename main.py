#!/usr/bin/env python3
"""Main entry point for Terrarium Agent."""

import asyncio
import sys
from pathlib import Path

from llm.vllm_client import VLLMClient
from agent.runtime import AgentRuntime
from agent.context import ContextManager
from tools.irc import IRCTool
from tools.harness import HarnessAdapter
from tools.harness_examples import NumberGuessHarness, TextAdventureHarness


async def run_harness_session(agent: AgentRuntime, tool_name: str):
    """
    Run an interactive harness session.

    Args:
        agent: AgentRuntime instance
        tool_name: Name of harness tool (e.g., "harness_number_guess")
    """
    print(f"\n{'='*60}")
    print(f"Starting harness: {tool_name}")
    print(f"{'='*60}\n")

    # Reset harness to start new episode
    result = await agent.execute_tool(tool_name, action="reset")

    if result.is_error():
        print(f"Error starting harness: {result.error}")
        return

    obs = result.output
    print(obs['content'])
    print()

    # Show available actions
    if obs.get('available_actions'):
        print("Available actions:")
        for action in obs['available_actions']:
            print(f"  - {action['name']}: {action['description']}")
        print()

    step = 1

    while not obs.get('done', False):
        try:
            # Get user input
            user_input = input(f"[Step {step}] Action (or 'quit' to exit): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting harness...")
                break

            if not user_input:
                # Just observe
                result = await agent.execute_tool(tool_name, action="observe")
                obs = result.output
                print(obs['content'])
                print()
                continue

            # Parse action and parameters
            # Format: "action_name param1=value1 param2=value2"
            parts = user_input.split()
            action_name = parts[0]
            params = {}

            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    # Try to convert to int if possible
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                    params[key] = value

            # Execute action
            result = await agent.execute_tool(
                tool_name,
                action="step",
                action_name=action_name,
                **params
            )

            if result.is_error():
                print(f"❌ Error: {result.error}\n")
                continue

            obs = result.output
            print()
            print(obs['content'])

            if result.metadata.get('reward'):
                print(f"\nReward: {result.metadata['reward']:+.1f}")

            print()
            step += 1

        except KeyboardInterrupt:
            print("\n\nExiting harness...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    # Get final stats
    result = await agent.execute_tool(tool_name, action="stats")
    if result.is_success():
        stats = result.output
        print(f"\n{'='*60}")
        print("Episode Statistics:")
        print(f"  Steps: {stats['total_steps']}")
        print(f"  Total Reward: {stats['total_reward']:.1f}")
        print(f"  Success: {'✓' if stats['success'] else '✗'}")
        print(f"  Duration: {stats['duration_seconds']:.2f}s")
        print(f"{'='*60}\n")


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

    # Register harnesses (wrapped as tools)
    print("\nRegistering harnesses...")
    harness_number_guess = HarnessAdapter(NumberGuessHarness())
    harness_text_adventure = HarnessAdapter(TextAdventureHarness())
    agent.register_tool(harness_number_guess)
    agent.register_tool(harness_text_adventure)
    print(f"  - {harness_number_guess.name}")
    print(f"  - {harness_text_adventure.name}")

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
    print("  /harness <name>  - Start a harness session")
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

                elif command == 'harness':
                    if len(parts) < 2:
                        print("Usage: /harness <name>")
                        print("Available harnesses:")
                        for tool_name in agent.tools.keys():
                            if tool_name.startswith("harness_"):
                                print(f"  - {tool_name.replace('harness_', '')}")
                        continue

                    harness_name = parts[1]
                    tool_name = f"harness_{harness_name}"

                    if tool_name not in agent.tools:
                        print(f"Harness not found: {harness_name}")
                        continue

                    await run_harness_session(agent, tool_name)

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
