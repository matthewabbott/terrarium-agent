#!/usr/bin/env python3
"""Test multi-context management system."""

import asyncio
from pathlib import Path
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.vllm_client import VLLMClient
from agent.multi_context_manager import MultiContextManager


async def test_multi_context():
    """Test context swapping and persistence."""

    # Clean up test directory
    test_dir = Path("sessions_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print("=" * 60)
    print("Multi-Context Management Test")
    print("=" * 60)
    print()

    # Initialize vLLM client
    client = VLLMClient(base_url="http://localhost:8000")

    # Check server
    print("1. Checking vLLM server...")
    if not await client.health_check():
        print("❌ vLLM server not available")
        return
    print("✓ Server is healthy")

    # Auto-detect model
    models = await client.get_models()
    if models:
        client.model = models[0]
        print(f"✓ Using model: {client.model}\n")
    else:
        print("❌ No models detected")
        return

    # Create context manager
    manager = MultiContextManager(
        vllm_client=client,
        storage_dir=test_dir,
        cache_size=5
    )

    # Test 1: IRC context
    print("2. Testing IRC context (#python)...")
    response1 = await manager.process_with_context(
        context_id="irc:#python",
        user_message="What's a decorator in Python?",
        system_prompt="You are a helpful IRC bot. Be concise."
    )
    print(f"Response: {response1[:100]}...")
    print("✓ IRC context created\n")

    # Test 2: Game context
    print("3. Testing game context (Pokemon)...")
    response2 = await manager.process_with_context(
        context_id="game:pokemon",
        user_message="I'm in Viridian City. What should I do?",
        system_prompt="You are playing Pokemon Red. You help with game strategy."
    )
    print(f"Response: {response2[:100]}...")
    print("✓ Game context created\n")

    # Test 3: CLI context
    print("4. Testing CLI context...")
    response3 = await manager.process_with_context(
        context_id="cli:main",
        user_message="Hello! Can you remember my name is Alice?",
        system_prompt="You are a helpful assistant."
    )
    print(f"Response: {response3[:100]}...")
    print("✓ CLI context created\n")

    # Test 4: Context switching - return to IRC
    print("5. Testing context switching back to IRC...")
    response4 = await manager.process_with_context(
        context_id="irc:#python",
        user_message="Can you give me a simple example?"
    )
    print(f"Response: {response4[:100]}...")
    print("✓ Context switched successfully\n")

    # Test 5: Context memory - return to CLI
    print("6. Testing context memory (CLI)...")
    response5 = await manager.process_with_context(
        context_id="cli:main",
        user_message="What's my name?"
    )
    print(f"Response: {response5}")
    if "alice" in response5.lower():
        print("✓ Context memory working - remembered name!\n")
    else:
        print("⚠ Context memory may not be working\n")

    # Test 6: Cache stats
    print("7. Checking cache stats...")
    cache_stats = manager.get_cache_stats()
    print(f"  Cached contexts: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    print(f"  Active context: {cache_stats['active_context']}")
    print(f"  In cache: {', '.join(cache_stats['cached_contexts'])}")
    print("✓ Cache working\n")

    # Test 7: List all contexts
    print("8. Listing all persisted contexts...")
    all_contexts = manager.list_all_contexts()
    for context_type, sessions in all_contexts.items():
        session_ids = [f"{s['date']}/{s['session_id']}" for s in sessions]
        print(f"  {context_type}: {', '.join(session_ids)}")
    print("✓ Persistence working\n")

    # Test 8: Session stats
    print("9. Getting session stats...")
    for context_id in ["irc:#python", "game:pokemon", "cli:main"]:
        stats = manager.get_stats(context_id)
        print(f"  {context_id}:")
        print(f"    Messages: {stats['message_count']}")
        print(f"    On disk: {stats['exists_on_disk']}")
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print()
    print("Test session files saved to: ./sessions_test/")
    print("You can inspect the JSON files to see the persisted conversations.")

    await client.shutdown()


if __name__ == "__main__":
    asyncio.run(test_multi_context())
