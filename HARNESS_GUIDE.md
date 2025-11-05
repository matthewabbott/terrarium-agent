

# Harness System Guide

This guide explains how to create and use harnesses in Terrarium Agent.

## What is a Harness?

A **harness** is an environment or game that the agent can interact with. Unlike **tools** (which are utilities like IRC or shell commands), harnesses represent structured environments with:

- **Observations**: What the agent can perceive
- **Action Space**: Available actions at each step
- **Rewards**: Optional scoring/feedback
- **Episodes**: Discrete rounds with reset capability

## Architecture

```
┌─────────────────────────────────────────────┐
│           Agent Runtime                     │
│  (manages LLM + tools + context)            │
└────────────┬────────────────────────────────┘
             │
             ├─── BaseTool (utilities)
             │    ├─ IRCTool
             │    ├─ ShellTool
             │    └─ FilesTool
             │
             └─── HarnessAdapter (wraps BaseHarness)
                  │
                  └─── BaseHarness (environments)
                       ├─ NumberGuessHarness
                       ├─ TextAdventureHarness
                       ├─ CodingChallengeHarness
                       └─ ... (your custom harnesses)
```

## Creating a Harness

### 1. Extend BaseHarness

```python
from tools.harness import (
    BaseHarness,
    Observation,
    ActionResult,
    ActionDefinition,
    HarnessStatus
)

class MyGameHarness(BaseHarness):
    def __init__(self):
        super().__init__(
            name="my_game",
            description="Description of your game",
            max_steps=100,  # Optional: max steps per episode
            timeout_seconds=300.0  # Optional: time limit
        )
        # Your game state variables
        self.score = 0
        self.game_state = {}

    async def initialize(self) -> bool:
        """Called once when harness is loaded."""
        # Set up any persistent resources
        return True

    async def reset(self) -> Observation:
        """Start a new episode."""
        # Reset game state
        self.score = 0
        self.current_step = 0
        self.status = HarnessStatus.ACTIVE

        # Return initial observation
        return Observation(
            content="Game started! Do something.",
            content_type="text",
            available_actions=self.get_action_space(),
            metadata={"score": self.score},
            done=False
        )

    async def step(self, action: str, **kwargs) -> ActionResult:
        """Execute an action."""
        if action == "my_action":
            # Process action
            reward = 10.0
            done = False

            # Update state
            self.current_step += 1
            self.score += reward

            obs = Observation(
                content="Action executed!",
                content_type="text",
                available_actions=self.get_action_space(),
                metadata={"score": self.score},
                done=done,
                reward=reward
            )

            return ActionResult(
                success=True,
                observation=obs,
                reward=reward
            )
        else:
            # Invalid action
            return ActionResult(
                success=False,
                observation=await self.get_observation(),
                error=f"Unknown action: {action}"
            )

    async def get_observation(self) -> Observation:
        """Get current state without taking action."""
        return Observation(
            content=f"Current score: {self.score}",
            content_type="text",
            available_actions=self.get_action_space(),
            metadata={"score": self.score},
            done=self.is_done()
        )

    def get_action_space(self) -> List[ActionDefinition]:
        """Define all possible actions."""
        return [
            ActionDefinition(
                name="my_action",
                description="Does something in the game",
                parameters={
                    "param1": {
                        "type": "string",
                        "description": "A parameter"
                    }
                }
            )
        ]

    async def shutdown(self):
        """Clean up resources."""
        pass
```

### 2. Register with Agent

```python
from tools.harness import HarnessAdapter
from tools.harness_examples import NumberGuessHarness

# Create harness
harness = NumberGuessHarness()

# Wrap in adapter (makes it compatible with tool system)
adapter = HarnessAdapter(harness)

# Register with agent runtime
agent.register_tool(adapter)
```

## Using Harnesses

### Through Agent Runtime

Once registered, the agent can interact with harnesses through the tool interface:

```python
# Reset harness (start new episode)
result = await agent.execute_tool(
    tool_name="harness_number_guess",
    action="reset"
)
# result.output contains the initial Observation

# Take action
result = await agent.execute_tool(
    tool_name="harness_number_guess",
    action="step",
    action_name="guess",
    number=50
)
# result.output contains new Observation
# result.metadata contains reward, done status

# Get current observation
result = await agent.execute_tool(
    tool_name="harness_number_guess",
    action="observe"
)

# Get episode statistics
result = await agent.execute_tool(
    tool_name="harness_number_guess",
    action="stats"
)
```

### Directly (for testing)

```python
# Create harness
harness = NumberGuessHarness()
await harness.initialize()

# Start episode
obs = await harness.reset()
print(obs.content)

# Take actions
while not obs.done:
    result = await harness.step("guess", number=50)
    obs = result.observation
    print(f"Reward: {result.reward}")
    print(obs.content)

# Get stats
stats = harness.get_stats()
print(f"Steps: {stats.total_steps}, Reward: {stats.total_reward}")
```

## Observation Format

Observations are what the agent perceives:

```python
Observation(
    content="Game state description or data",
    content_type="text",  # or "image", "json", etc.
    available_actions=[...],  # List of ActionDefinition
    metadata={},  # Extra info (score, time, etc.)
    done=False,  # Is episode complete?
    reward=10.0  # Optional reward signal
)
```

## Action Space

Actions define what the agent can do:

```python
ActionDefinition(
    name="move",
    description="Move in a direction",
    parameters={
        "direction": {
            "type": "string",
            "description": "Direction to move",
            "enum": ["north", "south", "east", "west"]
        }
    }
)
```

## Best Practices

### 1. Clear Observations
- Provide enough context for the agent to make decisions
- Include available actions in metadata
- Use descriptive error messages

### 2. Well-Defined Actions
- Keep action names simple and intuitive
- Validate parameters thoroughly
- Return informative errors for invalid actions

### 3. Episode Management
- Always reset state completely in `reset()`
- Track episode completion properly
- Set `done=True` when episode ends

### 4. Rewards (if using RL)
- Design rewards that guide desired behavior
- Use positive rewards for progress
- Small negative rewards for wasted actions
- Large rewards for achieving goals

### 5. State Persistence
- `initialize()` is called once - use for persistent resources
- `reset()` is called per episode - reset episode state
- `shutdown()` for cleanup

## Example: Testing a Harness

```python
import asyncio
from tools.harness_examples import NumberGuessHarness

async def test_harness():
    harness = NumberGuessHarness()
    await harness.initialize()

    # Start episode
    obs = await harness.reset()
    print(obs.content)
    print(f"Available actions: {[a.name for a in obs.available_actions]}")

    # Try to guess
    guesses = [50, 25, 75, 37, 62]
    for guess in guesses:
        result = await harness.step("guess", number=guess)
        print(f"\nGuess {guess}: {result.observation.content}")
        print(f"Reward: {result.reward}")

        if result.observation.done:
            break

    # Get final stats
    stats = harness.get_stats()
    print(f"\nFinal stats:")
    print(f"  Steps: {stats.total_steps}")
    print(f"  Reward: {stats.total_reward}")
    print(f"  Success: {stats.success}")

if __name__ == "__main__":
    asyncio.run(test_harness())
```

## Harness Ideas

- **Games**: Chess, tic-tac-toe, card games, puzzles
- **Coding**: LeetCode-style challenges, debugging tasks
- **CTF**: Security challenges, reverse engineering
- **Simulations**: Physics simulations, economic models
- **Creative**: Story generation, music composition
- **Research**: Scientific experiments, data analysis

## Advanced Features

### Multi-Agent Support

```python
class MultiAgentHarness(BaseHarness):
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.current_agent = 0

    async def step(self, action: str, **kwargs):
        # Process action for current agent
        # Switch to next agent
        self.current_agent = (self.current_agent + 1) % self.num_agents
        # ...
```

### Partial Observability

```python
async def get_observation(self) -> Observation:
    # Return different observations based on agent position
    visible_area = self.get_visible_area(self.agent_position)
    return Observation(content=visible_area, ...)
```

### Procedural Generation

```python
async def reset(self) -> Observation:
    # Generate new level/challenge each episode
    self.level = self.generate_random_level()
    return Observation(content=self.level.describe(), ...)
```

## Next Steps

1. Check out example harnesses in `tools/harness_examples.py`
2. Create your own harness by extending `BaseHarness`
3. Test it independently before integrating with agent
4. Register with agent using `HarnessAdapter`
5. Let the agent play!
