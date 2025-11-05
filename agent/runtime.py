"""Core agent runtime for executing tasks with tools and LLM."""

from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path

from llm.vllm_client import VLLMClient
from tools.base import BaseTool, ToolResult, ToolResultStatus
from .context import ContextManager, AgentContext


class AgentRuntime:
    """
    Core agent runtime that orchestrates LLM, tools, and context.

    The runtime:
    1. Receives input/task
    2. Loads appropriate context
    3. Calls LLM with context and available tools
    4. Executes tool calls if requested
    5. Returns results
    """

    def __init__(
        self,
        llm_client: VLLMClient,
        context_manager: Optional[ContextManager] = None
    ):
        """
        Initialize agent runtime.

        Args:
            llm_client: VLLMClient instance for LLM calls
            context_manager: Optional ContextManager (created if not provided)
        """
        self.llm = llm_client
        self.context_manager = context_manager or ContextManager()
        self.tools: Dict[str, BaseTool] = {}
        self.running = False

    async def initialize(self):
        """Initialize runtime components."""
        # Initialize LLM client
        await self.llm.initialize()

        # Load contexts
        self.context_manager.load_contexts()

        # Initialize all registered tools
        for tool in self.tools.values():
            await tool.initialize()

        self.running = True
        print("Agent runtime initialized")

    async def shutdown(self):
        """Shutdown runtime and cleanup resources."""
        self.running = False

        # Shutdown tools
        for tool in self.tools.values():
            await tool.shutdown()

        # Shutdown LLM client
        await self.llm.shutdown()

        print("Agent runtime shutdown")

    def register_tool(self, tool: BaseTool):
        """
        Register a tool with the runtime.

        Args:
            tool: BaseTool instance to register
        """
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            BaseTool if found, None otherwise
        """
        return self.tools.get(name)

    async def execute_tool(
        self,
        tool_name: str,
        action: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool action.

        Args:
            tool_name: Name of tool to use
            action: Action to perform
            **kwargs: Action parameters

        Returns:
            ToolResult from tool execution
        """
        tool = self.get_tool(tool_name)

        if not tool:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Tool '{tool_name}' not found"
            )

        if not tool.enabled:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Tool '{tool_name}' is disabled"
            )

        try:
            result = await tool.execute(action, **kwargs)
            return result
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Tool execution failed: {str(e)}"
            )

    def get_available_tools_for_context(self) -> List[BaseTool]:
        """
        Get tools available in current context.

        Returns:
            List of BaseTool instances available to current context
        """
        available_tool_names = self.context_manager.get_available_tools()
        return [
            self.tools[name]
            for name in available_tool_names
            if name in self.tools and self.tools[name].enabled
        ]

    async def process(
        self,
        prompt: str,
        context_name: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Process a prompt with LLM and tools.

        Args:
            prompt: User prompt/task
            context_name: Optional context to use (defaults to current)
            additional_context: Optional extra context to inject

        Returns:
            Agent's response
        """
        if not self.running:
            raise RuntimeError("Agent runtime not initialized. Call initialize() first.")

        # Switch context if specified
        if context_name:
            self.context_manager.switch_context(context_name)

        # Get system prompt from context
        system_prompt = self.context_manager.get_system_prompt()

        # Add additional context if provided
        if additional_context:
            prompt = f"{additional_context}\n\n{prompt}"

        # Get available tools for context
        available_tools = self.get_available_tools_for_context()

        # Simple generation without tool calling for now
        # TODO: Implement tool calling loop
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def process_with_tools(
        self,
        prompt: str,
        context_name: Optional[str] = None,
        max_iterations: int = 5
    ) -> str:
        """
        Process prompt with tool calling capability.

        Args:
            prompt: User prompt/task
            context_name: Optional context to use
            max_iterations: Maximum tool calling iterations

        Returns:
            Agent's final response
        """
        if not self.running:
            raise RuntimeError("Agent runtime not initialized. Call initialize() first.")

        # Switch context if specified
        if context_name:
            self.context_manager.switch_context(context_name)

        # Get system prompt from context
        system_prompt = self.context_manager.get_system_prompt()

        # Get available tools
        available_tools = self.get_available_tools_for_context()

        # Convert tools to OpenAI function format
        tool_definitions = self._tools_to_openai_format(available_tools)

        # Tool calling loop
        current_prompt = prompt
        for iteration in range(max_iterations):
            # Call LLM with tools
            result = await self.llm.generate_with_tools(
                prompt=current_prompt,
                tools=tool_definitions,
                system_prompt=system_prompt
            )

            # Check if LLM wants to call tools
            if not result.get("tool_calls"):
                # No more tool calls, return final response
                return result["text"]

            # Execute tool calls
            tool_results = []
            for tool_call in result["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                # Execute tool
                tool_result = await self.execute_tool(
                    tool_name=tool_name,
                    **arguments
                )

                tool_results.append({
                    "tool": tool_name,
                    "result": tool_result
                })

            # Prepare next prompt with tool results
            results_text = "\n".join([
                f"Tool {r['tool']}: {r['result'].output}"
                for r in tool_results
            ])
            current_prompt = f"{prompt}\n\nTool results:\n{results_text}\n\nContinue:"

        # Max iterations reached
        return "Maximum tool calling iterations reached. Task may be incomplete."

    def _tools_to_openai_format(self, tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """
        Convert tools to OpenAI function calling format.

        Args:
            tools: List of BaseTool instances

        Returns:
            List of tool definitions in OpenAI format
        """
        definitions = []

        for tool in tools:
            capabilities = tool.get_capabilities()

            for capability in capabilities:
                definitions.append({
                    "type": "function",
                    "function": {
                        "name": f"{tool.name}_{capability['action']}",
                        "description": capability['description'],
                        "parameters": capability.get('parameters', {})
                    }
                })

        return definitions
