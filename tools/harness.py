"""Base harness interface for game/task environments.

Harnesses provide structured environments that the agent can interact with,
such as games, coding challenges, CTF challenges, etc.

Unlike tools (which are utilities), harnesses represent environments with:
- Observations: What the agent can see/perceive
- Action space: Available actions in the current state
- Rewards/scoring: Optional feedback mechanism
- Episode management: Reset, termination conditions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import json


class HarnessStatus(Enum):
    """Status of harness execution."""
    ACTIVE = "active"           # Episode in progress
    COMPLETED = "completed"     # Episode finished successfully
    FAILED = "failed"          # Episode terminated with failure
    TIMEOUT = "timeout"        # Episode exceeded time limit
    ERROR = "error"            # Internal error occurred


@dataclass
class ActionDefinition:
    """Definition of an available action in the harness."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class Observation:
    """Observation from the harness environment.

    Observations represent what the agent can perceive at a given timestep.
    """
    # Main observation content (text, image data, structured data, etc.)
    content: Any

    # Type hint for the agent (text, image, json, etc.)
    content_type: str = "text"

    # Available actions in this state
    available_actions: List[ActionDefinition] = field(default_factory=list)

    # Optional metadata (score, time remaining, hints, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Whether episode is complete
    done: bool = False

    # Optional reward signal
    reward: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            "content": self.content,
            "content_type": self.content_type,
            "available_actions": [a.to_dict() for a in self.available_actions],
            "metadata": self.metadata,
            "done": self.done,
            "reward": self.reward
        }

    def to_json(self) -> str:
        """Convert observation to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ActionResult:
    """Result of executing an action in the harness."""
    # Whether action was executed successfully
    success: bool

    # New observation after action
    observation: Observation

    # Reward for this action (optional)
    reward: Optional[float] = None

    # Error message if action failed
    error: Optional[str] = None

    # Additional info about the action execution
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeStats:
    """Statistics for a completed episode."""
    total_steps: int
    total_reward: float
    success: bool
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseHarness(ABC):
    """
    Base class for harness environments.

    A harness provides a structured environment for the agent to interact with,
    such as a game, coding challenge, CTF, or other task.

    Lifecycle:
        1. initialize() - Set up resources
        2. reset() - Start new episode
        3. step(action) - Execute actions
        4. get_observation() - Get current state
        5. shutdown() - Clean up resources
    """

    def __init__(
        self,
        name: str,
        description: str,
        max_steps: Optional[int] = None,
        timeout_seconds: Optional[float] = None
    ):
        """
        Initialize harness.

        Args:
            name: Unique identifier for this harness
            description: Human-readable description
            max_steps: Maximum steps per episode (None = unlimited)
            timeout_seconds: Maximum time per episode (None = unlimited)
        """
        self.name = name
        self.description = description
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds

        # Episode state
        self.status = HarnessStatus.ACTIVE
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_start_time: Optional[float] = None

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize harness resources.

        Called once when harness is first loaded.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def reset(self) -> Observation:
        """
        Reset harness to initial state and start new episode.

        Returns:
            Initial observation for new episode
        """
        pass

    @abstractmethod
    async def step(self, action: str, **kwargs) -> ActionResult:
        """
        Execute an action in the harness.

        Args:
            action: Name of action to execute
            **kwargs: Action-specific parameters

        Returns:
            ActionResult with new observation and reward
        """
        pass

    @abstractmethod
    async def get_observation(self) -> Observation:
        """
        Get current observation without taking action.

        Returns:
            Current observation
        """
        pass

    @abstractmethod
    def get_action_space(self) -> List[ActionDefinition]:
        """
        Get complete action space for this harness.

        Returns:
            List of all possible actions (may be state-dependent)
        """
        pass

    async def shutdown(self):
        """Clean up harness resources. Override if needed."""
        pass

    def is_done(self) -> bool:
        """Check if current episode is complete."""
        return self.status != HarnessStatus.ACTIVE

    def get_stats(self) -> EpisodeStats:
        """
        Get statistics for current/completed episode.

        Returns:
            EpisodeStats with episode information
        """
        import time
        duration = 0.0
        if self.episode_start_time:
            duration = time.time() - self.episode_start_time

        return EpisodeStats(
            total_steps=self.current_step,
            total_reward=self.episode_reward,
            success=(self.status == HarnessStatus.COMPLETED),
            duration_seconds=duration
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}("
            f"name='{self.name}', "
            f"status={self.status.value}, "
            f"step={self.current_step})>"
        )


class HarnessAdapter:
    """
    Adapter that wraps a harness as a tool for the agent runtime.

    This allows harnesses to be registered as tools and used by the agent
    through the standard tool interface.
    """

    def __init__(self, harness: BaseHarness):
        """
        Initialize adapter with a harness.

        Args:
            harness: BaseHarness instance to wrap
        """
        self.harness = harness
        self.name = f"harness_{harness.name}"
        self.description = (
            f"Harness environment: {harness.description}. "
            f"Use 'reset' to start episode, 'step' to take actions, "
            f"'observe' to get current state."
        )
        self.enabled = True

    async def initialize(self) -> bool:
        """Initialize wrapped harness."""
        return await self.harness.initialize()

    async def shutdown(self):
        """Shutdown wrapped harness."""
        await self.harness.shutdown()

    async def execute(self, action: str, **kwargs):
        """
        Execute harness action through tool interface.

        Supported actions:
        - reset: Start new episode
        - step: Take action in environment
        - observe: Get current observation
        - stats: Get episode statistics
        """
        from tools.base import ToolResult, ToolResultStatus

        try:
            if action == "reset":
                obs = await self.harness.reset()
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output=obs.to_dict(),
                    metadata={"harness": self.harness.name}
                )

            elif action == "step":
                action_name = kwargs.pop("action_name", None)
                if not action_name:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        output=None,
                        error="Missing required parameter: action_name"
                    )

                result = await self.harness.step(action_name, **kwargs)

                if not result.success:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        output=result.observation.to_dict(),
                        error=result.error
                    )

                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output=result.observation.to_dict(),
                    metadata={
                        "reward": result.reward,
                        "done": result.observation.done,
                        **result.info
                    }
                )

            elif action == "observe":
                obs = await self.harness.get_observation()
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output=obs.to_dict()
                )

            elif action == "stats":
                stats = self.harness.get_stats()
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output=vars(stats)
                )

            else:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output=None,
                    error=f"Unknown harness action: {action}"
                )

        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Harness execution failed: {str(e)}"
            )

    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Get tool capabilities for this harness."""
        # Get harness-specific actions
        action_space = self.harness.get_action_space()

        capabilities = [
            {
                "action": "reset",
                "description": f"Reset {self.harness.name} and start new episode",
                "parameters": {}
            },
            {
                "action": "step",
                "description": f"Take an action in {self.harness.name}",
                "parameters": {
                    "action_name": {
                        "type": "string",
                        "description": f"Action to take. Available: {', '.join(a.name for a in action_space)}"
                    },
                    # Dynamic parameters based on action
                    "**kwargs": {
                        "type": "object",
                        "description": "Action-specific parameters"
                    }
                }
            },
            {
                "action": "observe",
                "description": f"Get current observation from {self.harness.name}",
                "parameters": {}
            },
            {
                "action": "stats",
                "description": f"Get episode statistics for {self.harness.name}",
                "parameters": {}
            }
        ]

        return capabilities
