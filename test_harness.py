#!/usr/bin/env python3
"""Test script for harnesses - demonstrates harness functionality without requiring LLM."""

import asyncio
from tools.harness_examples import NumberGuessHarness, TextAdventureHarness
from tools.harness import HarnessAdapter


async def test_number_guess():
    """Test the number guessing game harness."""
    print("="*60)
    print("Testing NumberGuessHarness")
    print("="*60)

    harness = NumberGuessHarness()
    await harness.initialize()

    # Start episode
    obs = await harness.reset()
    print(f"\n{obs.content}\n")
    print(f"Available actions: {[a.name for a in obs.available_actions]}")

    # Binary search strategy
    low, high = 1, 100
    step = 0

    while not obs.done and step < 10:
        guess = (low + high) // 2
        print(f"\n[Step {step + 1}] Guessing: {guess} (range: {low}-{high})")

        result = await harness.step("guess", number=guess)
        obs = result.observation

        print(result.observation.content)

        if result.observation.done:
            break

        # Update range based on feedback
        if "too low" in result.observation.content.lower():
            low = guess + 1
        elif "too high" in result.observation.content.lower():
            high = guess - 1

        step += 1

    # Get final stats
    stats = harness.get_stats()
    print(f"\n{'='*60}")
    print("Episode Stats:")
    print(f"  Steps: {stats.total_steps}")
    print(f"  Total Reward: {stats.total_reward:.1f}")
    print(f"  Success: {stats.success}")
    print(f"  Duration: {stats.duration_seconds:.2f}s")
    print(f"{'='*60}\n")

    await harness.shutdown()


async def test_text_adventure():
    """Test the text adventure harness."""
    print("="*60)
    print("Testing TextAdventureHarness")
    print("="*60)

    harness = TextAdventureHarness()
    await harness.initialize()

    # Start episode
    obs = await harness.reset()
    print(f"\n{obs.content}\n")

    # Scripted playthrough
    actions = [
        ("go", {"direction": "north"}),
        ("go", {"direction": "north"}),
        ("take", {"item": "key"}),
        ("go", {"direction": "south"}),
        ("go", {"direction": "east"}),
        ("take", {"item": "treasure"}),
    ]

    for action_name, params in actions:
        if obs.done:
            break

        print(f"\n>>> {action_name} {params}")
        result = await harness.step(action_name, **params)
        obs = result.observation

        print(f"{obs.content}")
        if result.reward:
            print(f"Reward: +{result.reward:.0f}")

    # Get final stats
    stats = harness.get_stats()
    print(f"\n{'='*60}")
    print("Episode Stats:")
    print(f"  Steps: {stats.total_steps}")
    print(f"  Total Reward: {stats.total_reward:.1f}")
    print(f"  Success: {stats.success}")
    print(f"  Duration: {stats.duration_seconds:.2f}s")
    print(f"{'='*60}\n")

    await harness.shutdown()


async def test_harness_adapter():
    """Test the harness adapter (wraps harness as tool)."""
    print("="*60)
    print("Testing HarnessAdapter")
    print("="*60)

    harness = NumberGuessHarness()
    adapter = HarnessAdapter(harness)

    await adapter.initialize()

    print(f"\nAdapter name: {adapter.name}")
    print(f"Description: {adapter.description}")
    print("\nCapabilities:")
    for cap in adapter.get_capabilities():
        print(f"  - {cap['action']}: {cap['description']}")

    # Use through adapter interface (simulates agent tool usage)
    print("\n" + "="*60)
    print("Using harness through adapter...")
    print("="*60)

    # Reset
    result = await adapter.execute("reset")
    print(f"\n[Reset] Status: {result.status.value}")
    obs = result.output
    print(f"Content: {obs['content']}")

    # Make a few guesses
    for guess in [50, 75, 87, 93]:
        result = await adapter.execute(
            "step",
            action_name="guess",
            number=guess
        )
        print(f"\n[Guess {guess}] Status: {result.status.value}")
        if result.metadata:
            print(f"Reward: {result.metadata.get('reward')}")
            print(f"Done: {result.metadata.get('done')}")

        obs = result.output
        print(f"Content: {obs['content'][:100]}...")

        if result.metadata.get('done'):
            break

    # Get stats
    result = await adapter.execute("stats")
    print(f"\n[Stats]")
    for key, value in result.output.items():
        print(f"  {key}: {value}")

    await adapter.shutdown()
    print("\n" + "="*60 + "\n")


async def main():
    """Run all tests."""
    print("\n🧪 Harness System Tests\n")

    try:
        await test_number_guess()
        await asyncio.sleep(1)

        await test_text_adventure()
        await asyncio.sleep(1)

        await test_harness_adapter()

        print("✅ All tests completed successfully!\n")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
