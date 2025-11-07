#!/usr/bin/env python3
"""Simple CLI chat interface for terrarium-agent."""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Optional

from llm.vllm_client import VLLMClient
from agent.multi_context_manager import MultiContextManager


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
        system_prompt: str = None,
        session_id: Optional[str] = None,
        storage_dir: Optional[Path] = None
    ):
        """
        Initialize chat interface.

        Args:
            vllm_url: vLLM server URL
            model: Model name (will auto-detect if not provided)
            system_prompt: System prompt for the conversation
            session_id: Optional session ID for persistent conversations
            storage_dir: Directory for session storage (default: ./sessions)
        """
        self.vllm_url = vllm_url
        self.model = model
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.storage_dir = storage_dir
        self.client = None
        self.context_manager = None

        # Use in-memory session if no session_id provided (backwards compatible)
        if not session_id:
            self.session = ChatSession(system_prompt)
        else:
            self.session = None  # Will use context_manager instead

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

        # Initialize context manager if using persistent sessions
        if self.session_id:
            self.context_manager = MultiContextManager(
                vllm_client=self.client,
                storage_dir=self.storage_dir
            )
            # Full context ID format: cli:{session_id}
            self.context_id = f"cli:{self.session_id}"
            print(f"✓ Using persistent session: {self.context_id}")

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
        try:
            # Use context manager if available (persistent sessions)
            if self.context_manager:
                response = await self.context_manager.process_with_context(
                    context_id=self.context_id,
                    user_message=user_message,
                    system_prompt=self.system_prompt
                )
                return response

            # Otherwise use in-memory session (backwards compatible)
            else:
                # Add user message to history
                self.session.add_message("user", user_message)

                # Get full conversation history (includes system prompt)
                messages = self.session.get_history()

                # Use chat method with full history for context-aware responses
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
        if self.context_manager:
            print("  /stats    - Show session statistics")
        print("  /quit     - Exit chat (or Ctrl+C)")
        print()

    def print_history(self):
        """Print conversation history."""
        print("\n" + "="*60)
        print("Conversation History")
        print("="*60)

        # Get messages from appropriate source
        if self.context_manager:
            session = self.context_manager.get_or_create_session(
                self.context_id,
                self.system_prompt
            )
            messages = session.get_history()
        else:
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
                        if self.context_manager:
                            self.context_manager.clear_context(self.context_id)
                        else:
                            self.session.clear()
                        print("✓ Conversation history cleared")
                    elif command == 'history':
                        self.print_history()
                    elif command == 'stats' and self.context_manager:
                        stats = self.context_manager.get_stats(self.context_id)
                        print("\nSession Statistics:")
                        for key, value in stats.items():
                            print(f"  {key}: {value}")
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
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID for persistent conversations (e.g., 'main', 'project_x')"
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        type=Path,
        help="Directory for session storage (default: ./sessions)"
    )

    args = parser.parse_args()

    # Create and run chat interface
    chat = ChatInterface(
        vllm_url=args.url,
        model=args.model,
        system_prompt=args.system,
        session_id=args.session_id,
        storage_dir=args.storage_dir
    )

    await chat.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
