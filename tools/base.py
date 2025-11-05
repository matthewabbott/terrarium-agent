"""Base tool interface for agent tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolResultStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """Result from tool execution."""
    status: ToolResultStatus
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def is_success(self) -> bool:
        """Check if tool execution was successful."""
        return self.status == ToolResultStatus.SUCCESS

    def is_error(self) -> bool:
        """Check if tool execution failed."""
        return self.status == ToolResultStatus.ERROR


class BaseTool(ABC):
    """
    Base class for agent tools.

    Tools provide specific capabilities to the agent (IRC, code execution, web access, etc.).
    Each tool must implement the execute method and define its capabilities.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize tool.

        Args:
            name: Tool identifier (e.g., 'irc', 'shell', 'python')
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description
        self.enabled = True

    @abstractmethod
    async def execute(self, action: str, **kwargs) -> ToolResult:
        """
        Execute a tool action.

        Args:
            action: The action to perform (tool-specific)
            **kwargs: Action-specific parameters

        Returns:
            ToolResult with status and output

        Example:
            # IRC tool
            result = await irc_tool.execute('send_message', channel='#test', text='Hello')

            # Shell tool
            result = await shell_tool.execute('run', command='ls -la')
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """
        Get list of capabilities this tool provides.

        Returns:
            List of capability definitions with action names, parameters, and descriptions

        Example:
            [
                {
                    'action': 'send_message',
                    'description': 'Send a message to IRC channel',
                    'parameters': {
                        'channel': 'str - Channel name',
                        'text': 'str - Message to send'
                    }
                }
            ]
        """
        pass

    async def initialize(self) -> bool:
        """
        Initialize tool resources (connections, clients, etc.).

        Returns:
            True if initialization successful, False otherwise
        """
        return True

    async def shutdown(self):
        """Cleanup tool resources."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})>"
