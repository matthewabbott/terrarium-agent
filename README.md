# Terrarium Agent

A flexible agent runtime with sandboxed execution, vLLM integration, and extensible tool system.

## Architecture

**Agent-first design** where tools (IRC, code execution, web, files) are integrated into a unified agent runtime, rather than building tool-specific bots.

### Components

- **HTTP API Server**: OpenAI-compatible REST API for external integration (port 8080)
- **Agent Runtime**: Core loop with context management and tool orchestration
- **vLLM Integration**: GLM-4.5-Air-AWQ-4bit via vLLM Docker container (port 8000)
- **Tool System**: Pluggable tools with clean interfaces
- **Context Swapping**: Different personas/contexts per domain (IRC ambassador, coder, etc.)
- **Session Management**: Persistent conversation storage with multi-context support
- **Harness System**: Structured game/task environments (chess, CTF, coding challenges)

## Project Structure

```
terrarium-agent/
├── agent/           # Core agent runtime
├── llm/             # vLLM client and prompt management
├── tools/           # Tool implementations (IRC, shell, python, files)
├── config/          # Configuration and context definitions
├── main.py          # Entry point
└── requirements.txt
```

## Tools

### IRC Tool
Integrates with `terrarium-irc` for reading/sending IRC messages, accessing chat history.

### Shell Tool
Execute shell commands in sandboxed environment.

### Python Tool
Execute Python code with resource limits.

### Files Tool
Read/write files with access controls.

## Setup

### Prerequisites

- **Docker** + **nvidia-container-toolkit** (for vLLM)
- **NVIDIA GPU:** GB10 (Blackwell) or compatible with driver 580+
- **Model:** GLM-4.5-Air-AWQ-4bit downloaded to `models/` directory

### Quick Start

```bash
# 1. Check model is downloaded
./check_model.sh

# 2. Install Python client dependencies
pip install -r requirements.txt

# 3. Start vLLM server in Docker (separate terminal)
./start_vllm_docker.sh

# 4. Choose how to run the agent:

# Option A: HTTP API Server (recommended for external integration)
source venv/bin/activate
python server.py  # Starts on http://localhost:8080

# Option B: Interactive chat with persistent sessions
source venv/bin/activate
python chat.py

# Option C: Full agent runtime with tools and harnesses
source venv/bin/activate
python main.py
```

**Documentation:**
- [QUICKSTART.md](QUICKSTART.md) - Detailed setup instructions
- [INTEGRATION.md](INTEGRATION.md) - **How to integrate external apps (IRC, web, games)**
- [AGENT_API.md](AGENT_API.md) - HTTP API specification
- [DOCKER_SETUP.md](DOCKER_SETUP.md) - Docker setup and troubleshooting

## Configuration

See `config/` directory for:
- `agent.yaml` - Main agent configuration
- `tools.yaml` - Tool-specific settings
- `contexts/` - Context definitions for different domains

## Development Status

🚧 **Early Development** - Core architecture being built

- [ ] vLLM client integration
- [ ] Base tool interface
- [ ] IRC tool (wrapping terrarium-irc)
- [ ] Agent runtime loop
- [ ] Context management
- [ ] Sandbox implementation

## External Integration

Terrarium Agent provides an **HTTP API** for integration with external applications.

**For IRC Integration:**
- See [INTEGRATION.md](INTEGRATION.md) for complete guide
- Start agent server: `python server.py`
- Make HTTP requests from terrarium-irc to `http://localhost:8080/v1/chat/completions`
- Client manages conversation history (stateless server)

**Other Use Cases:**
- Web chat applications
- Game environments (harnesses)
- Custom tools and bots

## Related Projects

- [terrarium-irc](../terrarium-irc) - IRC bot (integrates via HTTP API)