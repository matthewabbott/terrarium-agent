# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Terrarium Agent is a flexible agent runtime with sandboxed execution, vLLM integration, and an extensible tool/harness system. The agent-first architecture allows tools (IRC, code execution, web, files) to be integrated into a unified runtime rather than building tool-specific bots.

**Key Innovation:** Harness system - structured game/task environments (chess, CTF, coding challenges) that agents can interact with, separate from utility tools.

## vLLM Server Management

**NOTE:** vLLM now runs in Docker using NVIDIA's official container. This ensures compatibility with the GB10 GPU (Blackwell architecture, sm_12.1).

### Starting the vLLM Server
```bash
./start_vllm_docker.sh
```
This starts vLLM in a Docker container with GLM-4.5-Air-AWQ-4bit (4-bit quantized model, ~60GB). The server loads on port 8000 with tensor parallelism support and GLM-4.5 tool calling/reasoning parsers enabled.

**Container details:**
- Image: `nvcr.io/nvidia/vllm:25.09-py3` (NVIDIA official)
- vLLM version: 0.10.1.1
- Includes proper GB10/Blackwell support
- Pre-configured with CUDA 12.8+ libraries

### Checking Model Download Status
```bash
./check_model.sh
```
Verifies that all 13 safetensors files and required configs are present in `models/GLM-4.5-Air-AWQ-4bit/`.

### Managing the Docker Container
```bash
# Check logs
docker logs -f vllm-server

# Stop server
docker stop vllm-server

# Restart server
./start_vllm_docker.sh

# Health check
curl http://localhost:8000/health
```

### Running the Agent
```bash
# Interactive chat (simple interface)
source venv/bin/activate
python chat.py

# Full agent runtime with tools and harnesses
python main.py
```

## Architecture

### Core Components

**AgentRuntime** (`agent/runtime.py`) - Central orchestrator that manages:
- LLM client lifecycle
- Tool/harness registration and execution
- Context switching between personas
- Tool calling loop (planned feature)

**ContextManager** (`agent/context.py`) - Manages agent contexts/personas:
- Loads contexts from `config/contexts/*.yaml`
- Each context defines: system prompt, available tools, personality traits, constraints
- Examples: IRC ambassador (friendly, concise) vs Coder (technical, detailed)

**VLLMClient** (`llm/vllm_client.py`) - OpenAI-compatible client for vLLM:
- `generate()` - Simple text generation with system prompt
- `generate_with_tools()` - Tool calling capability (GLM-4.5 native support)
- `health_check()` - Verify server availability

### Tool System

**BaseTool** (`tools/base.py`) - Abstract base for utility tools:
- `execute(action, **kwargs)` - Perform tool action
- `get_capabilities()` - Define available actions
- Returns `ToolResult` with status/output/error

**IRCTool** (`tools/irc.py`) - IRC integration (currently mock, designed for terrarium-irc integration)

### Harness System

**BaseHarness** (`tools/harness.py`) - Abstract base for game/task environments:
- `reset()` - Start new episode, returns initial `Observation`
- `step(action, **kwargs)` - Execute action, returns `ActionResult`
- `get_observation()` - Get current state without taking action
- `get_action_space()` - Define available actions as `ActionDefinition` list
- Episode lifecycle: `initialize() → reset() → step()* → get_stats()`

**HarnessAdapter** - Wraps harnesses to be compatible with the tool system (agent sees them as tools)

**Observation** - What agent perceives: content, available actions, metadata, done flag, optional reward

**Key Distinction:** Tools are utilities (IRC, shell, files). Harnesses are structured environments with observation/action/reward/episodes (games, CTF, coding challenges).

### Integration with terrarium-irc

Design follows library integration pattern (see `INTEGRATION_DESIGN.md`):
1. terrarium-irc imports terrarium-agent as library
2. Uses `AgentClient` class (in `llm/` - planned) for on-demand agent invocation
3. Agent runtime created per query or kept warm
4. vLLM server runs persistently
5. IRC bot passes channel context to agent

## Key Files

- `agent/runtime.py` - Core agent loop and tool execution
- `agent/context.py` - Context/persona management
- `llm/vllm_client.py` - vLLM API client
- `tools/base.py` - Tool interface
- `tools/harness.py` - Harness (environment) interface
- `tools/harness_examples.py` - Example harnesses (NumberGuess, TextAdventure)
- `main.py` - Interactive agent with harness sessions
- `chat.py` - Simple CLI chat interface
- `config/contexts/*.yaml` - Context definitions

## Development Patterns

### Adding a New Tool
1. Create `tools/your_tool.py` extending `BaseTool`
2. Implement `execute(action, **kwargs)` and `get_capabilities()`
3. Register in `main.py`: `agent.register_tool(YourTool())`
4. Add to context YAML: `available_tools: [your_tool]`

### Adding a New Harness
1. Create harness extending `BaseHarness` in `tools/` or separate file
2. Implement: `initialize()`, `reset()`, `step()`, `get_observation()`, `get_action_space()`
3. Wrap with `HarnessAdapter`: `adapter = HarnessAdapter(YourHarness())`
4. Register: `agent.register_tool(adapter)`
5. Access via `/harness your_name` in interactive mode

### Adding a New Context
1. Create `config/contexts/your_context.yaml`
2. Define: name, description, system_prompt, available_tools, personality_traits, constraints
3. Auto-loaded on runtime initialization
4. Switch via: `agent.context_manager.switch_context("your_context")`

## Dependencies

**vLLM (Docker):**
- `docker` - Container runtime
- `nvidia-container-toolkit` - GPU access in containers
- NVIDIA vLLM image: `nvcr.io/nvidia/vllm:25.09-py3`
  - Includes vLLM 0.10.1.1
  - Includes PyTorch with GB10/Blackwell support
  - Includes CUDA 12.8+ libraries

**Python venv (Client only):**
- `openai>=1.50.0` - Client library for vLLM's OpenAI-compatible API
- `aiohttp>=3.9.0` - Async HTTP for API calls
- `pyyaml>=6.0` - Context file parsing
- `pydantic>=2.0.0` - Data validation

**Optional:**
- `-e ../terrarium-irc` - Local package for IRC tool integration

**NOTE:** PyTorch and vLLM are NOT in the venv - they run in the Docker container. The venv only contains lightweight client libraries.

## Testing

Test a harness directly:
```python
harness = NumberGuessHarness()
await harness.initialize()
obs = await harness.reset()
result = await harness.step("guess", number=50)
```

Interactive harness session:
```bash
python main.py
# Then: /harness number_guess
```

## Important Notes

- **Tool calling is partially implemented** - `process_with_tools()` exists but not fully integrated
- **IRC tool is mock** - Returns placeholder data until terrarium-irc integration complete
- **Harness adapter** makes harnesses look like tools to the runtime (unified interface)
- **Context swapping** allows same agent to have different personas (IRC friendly vs coding expert)
- **vLLM must be running** before starting agent - check with `curl http://localhost:8000/health`
- **Model loading takes time** - First vLLM startup loads ~60GB into memory

## Future Work

From README and design docs:
- Complete tool calling loop in `AgentRuntime.process_with_tools()`
- Implement real IRC tool using terrarium-irc client
- Add Python execution tool (sandbox)
- Add shell tool (safe command execution)
- Add file operations tool
- Create `AgentClient` class for external invocation
- Optional HTTP API for agent runtime
