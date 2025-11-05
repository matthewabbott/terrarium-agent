"""Context management for agent personalities and tool configurations."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path


@dataclass
class AgentContext:
    """
    Agent context defining personality, capabilities, and tool availability.

    Different contexts allow the agent to have different personas:
    - IRC ambassador: Friendly, conversational, has IRC tool
    - Coder: Technical, has code execution and file tools
    - Researcher: Analytical, has web search and file tools
    """

    name: str
    description: str
    system_prompt: str
    available_tools: List[str]
    personality_traits: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContext':
        """Create context from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            system_prompt=data['system_prompt'],
            available_tools=data.get('available_tools', []),
            personality_traits=data.get('personality_traits', {}),
            constraints=data.get('constraints', {}),
            metadata=data.get('metadata', {})
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'AgentContext':
        """Load context from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'system_prompt': self.system_prompt,
            'available_tools': self.available_tools,
            'personality_traits': self.personality_traits,
            'constraints': self.constraints,
            'metadata': self.metadata
        }


class ContextManager:
    """
    Manages agent contexts and handles switching between them.

    Contexts are swapped in based on the current task/domain:
    - IRC message? Load IRC context
    - Code request? Load coder context
    - Research task? Load researcher context
    """

    def __init__(self, contexts_dir: Optional[Path] = None):
        """
        Initialize context manager.

        Args:
            contexts_dir: Directory containing context YAML files
        """
        self.contexts_dir = contexts_dir or Path("config/contexts")
        self.contexts: Dict[str, AgentContext] = {}
        self.current_context: Optional[AgentContext] = None
        self.default_context_name: Optional[str] = None

    def load_contexts(self):
        """Load all context definitions from contexts directory."""
        if not self.contexts_dir.exists():
            print(f"Warning: Contexts directory not found: {self.contexts_dir}")
            return

        for yaml_file in self.contexts_dir.glob("*.yaml"):
            try:
                context = AgentContext.from_yaml(yaml_file)
                self.contexts[context.name] = context
                print(f"Loaded context: {context.name}")

                # Set first context as default if not set
                if self.default_context_name is None:
                    self.default_context_name = context.name

            except Exception as e:
                print(f"Error loading context from {yaml_file}: {e}")

    def add_context(self, context: AgentContext):
        """
        Add a context programmatically.

        Args:
            context: AgentContext to add
        """
        self.contexts[context.name] = context

        if self.default_context_name is None:
            self.default_context_name = context.name

    def get_context(self, name: str) -> Optional[AgentContext]:
        """
        Get context by name.

        Args:
            name: Context name

        Returns:
            AgentContext if found, None otherwise
        """
        return self.contexts.get(name)

    def switch_context(self, name: str) -> bool:
        """
        Switch to a different context.

        Args:
            name: Name of context to switch to

        Returns:
            True if switch successful, False if context not found
        """
        context = self.contexts.get(name)
        if context:
            self.current_context = context
            print(f"Switched to context: {name}")
            return True
        else:
            print(f"Context not found: {name}")
            return False

    def get_current_context(self) -> Optional[AgentContext]:
        """
        Get current active context.

        Returns:
            Current AgentContext, or default if none set
        """
        if self.current_context:
            return self.current_context

        # Load default context if available
        if self.default_context_name:
            self.switch_context(self.default_context_name)
            return self.current_context

        return None

    def list_contexts(self) -> List[str]:
        """
        Get list of available context names.

        Returns:
            List of context names
        """
        return list(self.contexts.keys())

    def get_available_tools(self) -> List[str]:
        """
        Get list of tools available in current context.

        Returns:
            List of tool names available to current context
        """
        context = self.get_current_context()
        return context.available_tools if context else []

    def get_system_prompt(self) -> str:
        """
        Get system prompt for current context.

        Returns:
            System prompt string, or empty string if no context
        """
        context = self.get_current_context()
        return context.system_prompt if context else ""
