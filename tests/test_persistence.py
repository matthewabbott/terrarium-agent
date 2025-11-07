#!/usr/bin/env python3
"""Test session persistence across 'restarts'."""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.vllm_client import VLLMClient
from agent.multi_context_manager import MultiContextManager


async def test_persistence():
    """Test that sessions persist and resume correctly."""

    test_dir = Path("sessions_test")

    print("=" * 60)
    print("Session Persistence Test")
    print("=" * 60)
    print()

    # Initialize vLLM client
    client = VLLMClient(base_url="http://localhost:8000")

    if not await client.health_check():
        print("❌ vLLM server not available")
        return

    # Auto-detect model
    models = await client.get_models()
    if models:
        client.model = models[0]

    # Create FIRST manager instance (simulates first run)
    print("1. Creating first manager instance...")
    manager1 = MultiContextManager(
        vllm_client=client,
        storage_dir=test_dir,
        cache_size=5
    )

    # Use the CLI context that was created in previous test
    print("2. Accessing existing CLI context from previous test...")
    response1 = await manager1.process_with_context(
        context_id="cli:main",
        user_message="Do you still remember my name?"
    )
    print(f"Response: {response1}")

    if "alice" in response1.lower():
        print("✓ Session loaded from disk - remembered Alice!\n")
    else:
        print("⚠ Session may not have loaded correctly\n")

    # Check stats
    stats1 = manager1.get_stats("cli:main")
    message_count_1 = stats1['message_count']
    print(f"3. Message count in session: {message_count_1}\n")

    # Delete manager1 (simulates program exit)
    print("4. Deleting first manager (simulating program exit)...")
    del manager1
    print("✓ First manager deleted\n")

    # Create SECOND manager instance (simulates restart)
    print("5. Creating second manager instance (simulating restart)...")
    manager2 = MultiContextManager(
        vllm_client=client,
        storage_dir=test_dir,
        cache_size=5
    )
    print("✓ Second manager created\n")

    # Access the same context again
    print("6. Accessing CLI context from second manager...")
    response2 = await manager2.process_with_context(
        context_id="cli:main",
        user_message="What was my name again?"
    )
    print(f"Response: {response2}")

    if "alice" in response2.lower():
        print("✓ Session persisted across restart - still remembers Alice!\n")
    else:
        print("⚠ Session may not have persisted\n")

    # Check stats again
    stats2 = manager2.get_stats("cli:main")
    message_count_2 = stats2['message_count']
    print(f"7. Message count after restart: {message_count_2}")
    print(f"   (Should be {message_count_1 + 2} if persistence working)\n")

    if message_count_2 == message_count_1 + 2:
        print("✓ Message count correct - full history persisted!\n")
    else:
        print(f"⚠ Message count mismatch: expected {message_count_1 + 2}, got {message_count_2}\n")

    # Verify file exists
    session_file = test_dir / "cli" / "main.json"
    print(f"8. Checking if session file exists...")
    print(f"   File: {session_file}")
    print(f"   Exists: {session_file.exists()}")
    if session_file.exists():
        print(f"   Size: {session_file.stat().st_size} bytes")
        print("✓ Session file persisted to disk\n")

    print("=" * 60)
    print("Persistence test completed! ✓")
    print("=" * 60)

    await client.shutdown()


if __name__ == "__main__":
    asyncio.run(test_persistence())
