#!/usr/bin/env python3
"""Simple CLI chat interface for terrarium-agent."""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict

from llm.vllm_client import VLLMClient


class ChatSession:
    """Manages a chat session with conversation history."""

    def __init__(self, system_prompt: str = None):
        """
        Initialize chat session.

        Args:
            system_prompt: System prompt for the conversation
        """
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = system_prompt or "You are a helpful AI assistant."

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Get full conversation history including system prompt."""
        history = [{"role": "system", "content": self.system_prompt}]
        history.extend(self.messages)
        return history

    def clear(self):
        """Clear conversation history."""
        self.messages = []

    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """Get last n messages from history."""
        return self.messages[-n:] if n > 0 else self.messages


class ChatInterface:
    """Interactive CLI chat interface."""

    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        model: str = None,
        system_prompt: str = None
    ):
        """
        Initialize chat interface.

        Args:
            vllm_url: vLLM server URL
            model: Model name (will auto-detect if not provided)
            system_prompt: System prompt for the conversation
        """
        self.vllm_url = vllm_url
        self.model = model
        self.client = None
        self.session = ChatSession(system_prompt)

    async def initialize(self):
        """Initialize vLLM client and check server health."""
        print("Initializing chat interface...")

        # Create client (model will be set after detection)
        self.client = VLLMClient(base_url=self.vllm_url)

        # Check server health
        print(f"Connecting to vLLM server at {self.vllm_url}...")
        is_healthy = await self.client.health_check()

        if not is_healthy:
            print(f"\n❌ ERROR: vLLM server not responding at {self.vllm_url}")
            print("\nMake sure vLLM is running:")
            print("  ./start_vllm_docker.sh")
            sys.exit(1)

        print("✓ Server is healthy")

        # Get available models
        models = await self.client.get_models()

        if not models:
            print("⚠️  Warning: No models detected")
            self.model = "default"
        elif self.model:
            print(f"✓ Using model: {self.model}")
        else:
            # Auto-select first model
            self.model = models[0]
            print(f"✓ Auto-selected model: {self.model}")

        if len(models) > 1:
            print(f"  Available models: {', '.join(models)}")

        # Update client with detected model
        self.client.model = self.model

    async def shutdown(self):
        """Cleanup resources."""
        if self.client:
            await self.client.shutdown()

    async def generate_response(self, user_message: str) -> str:
        """
        Generate response to user message with full conversation history.

        Args:
            user_message: User's input

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.session.add_message("user", user_message)

        # Get full conversation history (includes system prompt)
        messages = self.session.get_history()

        # Use chat method with full history for context-aware responses
        try:
            response = await self.client.chat(messages=messages)

            # Add response to history
            self.session.add_message("assistant", response)

            return response

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"\n❌ {error_msg}")
            return None

    def print_help(self):
        """Print help message."""
        print("\nAvailable commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history")
        print("  /history  - Show conversation history")
        print("  /system   - Change system prompt")
        print("  /quit     - Exit chat (or Ctrl+C)")
        print()

    def print_history(self):
        """Print conversation history."""
        print("\n" + "="*60)
        print("Conversation History")
        print("="*60)

        messages = self.session.get_history()

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                print(f"\n[SYSTEM]")
                print(content)
            elif role == "user":
                print(f"\n[YOU]")
                print(content)
            elif role == "assistant":
                print(f"\n[ASSISTANT]")
                print(content)

        print("\n" + "="*60 + "\n")

    async def run(self):
        """Run interactive chat loop."""
        # Initialize
        await self.initialize()

        print("\n" + "="*60)
        print("Terrarium Agent - Chat Interface")
        print("="*60)
        print(f"\nConnected to: {self.vllm_url}")
        print(f"Model: {self.model}")
        print("\nType /help for commands, /quit to exit")
        print("="*60 + "\n")

        try:
            while True:
                # Get user input
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    # Handle Ctrl+D
                    print()
                    break

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()

                    if command in ['quit', 'exit', 'q']:
                        break
                    elif command in ['help', 'h']:
                        self.print_help()
                    elif command == 'clear':
                        self.session.clear()
                        print("✓ Conversation history cleared")
                    elif command == 'history':
                        self.print_history()
                    elif command == 'system':
                        new_prompt = input("Enter new system prompt: ").strip()
                        if new_prompt:
                            self.session.system_prompt = new_prompt
                            self.session.clear()  # Clear history when changing system
                            print("✓ System prompt updated and history cleared")
                    else:
                        print(f"Unknown command: /{command}")
                        print("Type /help for available commands")

                    continue

                # Generate response
                print("Assistant: ", end="", flush=True)
                response = await self.generate_response(user_input)

                if response:
                    print(response)
                    print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
        finally:
            await self.shutdown()
            print("Chat session ended.")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Terrarium Agent Chat Interface")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt (default: helpful assistant)"
    )

    args = parser.parse_args()

    # Create and run chat interface
    chat = ChatInterface(
        vllm_url=args.url,
        model=args.model,
        system_prompt=args.system
    )

    await chat.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
