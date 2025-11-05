"""Example harness implementations demonstrating the harness interface."""

import random
import time
from typing import List

from tools.harness import (
    BaseHarness,
    Observation,
    ActionResult,
    ActionDefinition,
    HarnessStatus
)


class NumberGuessHarness(BaseHarness):
    """
    Simple number guessing game harness.

    The agent needs to guess a secret number between 1 and 100.
    Provides feedback: "too high", "too low", or "correct".

    Good for testing basic harness functionality and agent reasoning.
    """

    def __init__(self):
        super().__init__(
            name="number_guess",
            description="Guess a secret number between 1 and 100",
            max_steps=10,  # Agent gets 10 guesses
            timeout_seconds=60.0
        )
        self.secret_number = 0
        self.guesses: List[int] = []

    async def initialize(self) -> bool:
        """Initialize game."""
        return True

    async def reset(self) -> Observation:
        """Start new game with new secret number."""
        self.secret_number = random.randint(1, 100)
        self.guesses = []
        self.current_step = 0
        self.episode_reward = 0.0
        self.status = HarnessStatus.ACTIVE
        self.episode_start_time = time.time()

        return Observation(
            content=(
                "Welcome to Number Guess!\n"
                "I'm thinking of a number between 1 and 100.\n"
                f"You have {self.max_steps} guesses.\n"
                "Good luck!"
            ),
            content_type="text",
            available_actions=self.get_action_space(),
            metadata={
                "guesses_remaining": self.max_steps,
                "guesses_made": []
            },
            done=False
        )

    async def step(self, action: str, **kwargs) -> ActionResult:
        """Execute a guess action."""
        if self.status != HarnessStatus.ACTIVE:
            return ActionResult(
                success=False,
                observation=await self.get_observation(),
                error="Game is not active. Use reset to start new game."
            )

        if action != "guess":
            return ActionResult(
                success=False,
                observation=await self.get_observation(),
                error=f"Unknown action: {action}. Use 'guess'."
            )

        # Get guess from kwargs
        guess = kwargs.get("number")
        if guess is None:
            return ActionResult(
                success=False,
                observation=await self.get_observation(),
                error="Missing required parameter: number"
            )

        try:
            guess = int(guess)
        except (ValueError, TypeError):
            return ActionResult(
                success=False,
                observation=await self.get_observation(),
                error=f"Invalid guess: {guess}. Must be an integer."
            )

        if guess < 1 or guess > 100:
            return ActionResult(
                success=False,
                observation=await self.get_observation(),
                error="Guess must be between 1 and 100"
            )

        # Process guess
        self.current_step += 1
        self.guesses.append(guess)

        # Check if correct
        if guess == self.secret_number:
            self.status = HarnessStatus.COMPLETED
            reward = 100.0 - (self.current_step * 5)  # Bonus for fewer guesses
            self.episode_reward += reward

            obs = Observation(
                content=(
                    f"🎉 Correct! The number was {self.secret_number}!\n"
                    f"You guessed it in {self.current_step} tries.\n"
                    f"Score: {reward:.1f}"
                ),
                content_type="text",
                available_actions=[],
                metadata={
                    "guesses": self.guesses,
                    "final_score": reward
                },
                done=True,
                reward=reward
            )

            return ActionResult(
                success=True,
                observation=obs,
                reward=reward
            )

        # Provide hint
        if guess < self.secret_number:
            hint = "📈 Too low! Try a higher number."
            reward = -1.0
        else:
            hint = "📉 Too high! Try a lower number."
            reward = -1.0

        self.episode_reward += reward

        # Check if out of guesses
        guesses_remaining = self.max_steps - self.current_step
        if guesses_remaining == 0:
            self.status = HarnessStatus.FAILED
            obs = Observation(
                content=(
                    f"{hint}\n"
                    f"💀 Game over! You ran out of guesses.\n"
                    f"The number was {self.secret_number}.\n"
                    f"Your guesses: {self.guesses}"
                ),
                content_type="text",
                available_actions=[],
                metadata={
                    "guesses": self.guesses,
                    "secret_number": self.secret_number
                },
                done=True,
                reward=reward
            )
        else:
            obs = Observation(
                content=(
                    f"{hint}\n"
                    f"Guesses remaining: {guesses_remaining}\n"
                    f"Your previous guesses: {self.guesses}"
                ),
                content_type="text",
                available_actions=self.get_action_space(),
                metadata={
                    "guesses": self.guesses,
                    "guesses_remaining": guesses_remaining
                },
                done=False,
                reward=reward
            )

        return ActionResult(
            success=True,
            observation=obs,
            reward=reward
        )

    async def get_observation(self) -> Observation:
        """Get current game state."""
        if self.status != HarnessStatus.ACTIVE:
            return Observation(
                content=f"Game is {self.status.value}. Use reset to start new game.",
                content_type="text",
                available_actions=[],
                done=True
            )

        guesses_remaining = self.max_steps - self.current_step
        return Observation(
            content=(
                f"Number Guess Game (Active)\n"
                f"Guesses remaining: {guesses_remaining}\n"
                f"Previous guesses: {self.guesses}"
            ),
            content_type="text",
            available_actions=self.get_action_space(),
            metadata={
                "guesses": self.guesses,
                "guesses_remaining": guesses_remaining
            },
            done=False
        )

    def get_action_space(self) -> List[ActionDefinition]:
        """Get available actions."""
        return [
            ActionDefinition(
                name="guess",
                description="Guess a number between 1 and 100",
                parameters={
                    "number": {
                        "type": "integer",
                        "description": "Your guess (1-100)",
                        "minimum": 1,
                        "maximum": 100
                    }
                }
            )
        ]


class TextAdventureHarness(BaseHarness):
    """
    Simple text adventure harness.

    Demonstrates a more complex harness with multiple rooms, items, and state.
    Good for testing agent navigation and exploration.
    """

    def __init__(self):
        super().__init__(
            name="text_adventure",
            description="Navigate a text adventure game",
            max_steps=50
        )

        # Game world
        self.rooms = {
            "start": {
                "description": "You are in a small room. There's a door to the NORTH.",
                "exits": {"north": "hallway"},
                "items": []
            },
            "hallway": {
                "description": "A long hallway. Doors lead NORTH, SOUTH, and EAST.",
                "exits": {"south": "start", "north": "library", "east": "treasure"},
                "items": []
            },
            "library": {
                "description": "A dusty library filled with old books.",
                "exits": {"south": "hallway"},
                "items": ["key"]
            },
            "treasure": {
                "description": "A treasure room! But it's locked...",
                "exits": {"west": "hallway"},
                "items": ["treasure"],
                "locked": True
            }
        }

        # Game state
        self.current_room = "start"
        self.inventory: List[str] = []

    async def initialize(self) -> bool:
        """Initialize adventure."""
        return True

    async def reset(self) -> Observation:
        """Start new adventure."""
        self.current_room = "start"
        self.inventory = []
        self.current_step = 0
        self.episode_reward = 0.0
        self.status = HarnessStatus.ACTIVE
        self.episode_start_time = time.time()

        # Reset room states
        self.rooms["library"]["items"] = ["key"]
        self.rooms["treasure"]["items"] = ["treasure"]
        self.rooms["treasure"]["locked"] = True

        room = self.rooms[self.current_room]
        return Observation(
            content=(
                "🗺️  Text Adventure\n\n"
                f"{room['description']}\n"
                f"Exits: {', '.join(room['exits'].keys())}\n"
                f"Items: {', '.join(room['items']) if room['items'] else 'none'}"
            ),
            content_type="text",
            available_actions=self.get_action_space(),
            metadata={
                "room": self.current_room,
                "inventory": self.inventory
            },
            done=False
        )

    async def step(self, action: str, **kwargs) -> ActionResult:
        """Execute adventure action."""
        if self.status != HarnessStatus.ACTIVE:
            return ActionResult(
                success=False,
                observation=await self.get_observation(),
                error="Adventure is not active."
            )

        self.current_step += 1
        room = self.rooms[self.current_room]

        if action == "go":
            direction = kwargs.get("direction", "").lower()
            if direction not in room["exits"]:
                return ActionResult(
                    success=False,
                    observation=await self.get_observation(),
                    error=f"Can't go {direction} from here."
                )

            next_room = room["exits"][direction]

            # Check if locked
            if self.rooms[next_room].get("locked", False):
                if "key" not in self.inventory:
                    return ActionResult(
                        success=False,
                        observation=await self.get_observation(),
                        error="The door is locked. You need a key!"
                    )
                else:
                    self.rooms[next_room]["locked"] = False

            self.current_room = next_room
            reward = 5.0  # Small reward for exploration

        elif action == "take":
            item = kwargs.get("item", "").lower()
            if item not in room["items"]:
                return ActionResult(
                    success=False,
                    observation=await self.get_observation(),
                    error=f"There's no {item} here."
                )

            room["items"].remove(item)
            self.inventory.append(item)
            reward = 10.0

            # Check win condition
            if item == "treasure":
                self.status = HarnessStatus.COMPLETED
                reward = 100.0

        elif action == "inventory":
            obs = await self.get_observation()
            obs.content += f"\n\n🎒 Inventory: {', '.join(self.inventory) if self.inventory else 'empty'}"
            return ActionResult(
                success=True,
                observation=obs,
                reward=0.0
            )

        else:
            return ActionResult(
                success=False,
                observation=await self.get_observation(),
                error=f"Unknown action: {action}"
            )

        self.episode_reward += reward
        obs = await self.get_observation()
        obs.reward = reward

        if self.status == HarnessStatus.COMPLETED:
            obs.content += "\n\n🏆 You found the treasure! YOU WIN!"
            obs.done = True

        return ActionResult(
            success=True,
            observation=obs,
            reward=reward
        )

    async def get_observation(self) -> Observation:
        """Get current adventure state."""
        room = self.rooms[self.current_room]

        content = (
            f"{room['description']}\n\n"
            f"Exits: {', '.join(room['exits'].keys())}\n"
            f"Items: {', '.join(room['items']) if room['items'] else 'none'}"
        )

        return Observation(
            content=content,
            content_type="text",
            available_actions=self.get_action_space(),
            metadata={
                "room": self.current_room,
                "inventory": self.inventory,
                "step": self.current_step
            },
            done=(self.status != HarnessStatus.ACTIVE)
        )

    def get_action_space(self) -> List[ActionDefinition]:
        """Get available actions."""
        return [
            ActionDefinition(
                name="go",
                description="Move in a direction",
                parameters={
                    "direction": {
                        "type": "string",
                        "description": "Direction to move (north, south, east, west)",
                        "enum": ["north", "south", "east", "west"]
                    }
                }
            ),
            ActionDefinition(
                name="take",
                description="Pick up an item",
                parameters={
                    "item": {
                        "type": "string",
                        "description": "Item to pick up"
                    }
                }
            ),
            ActionDefinition(
                name="inventory",
                description="Check your inventory",
                parameters={}
            )
        ]
