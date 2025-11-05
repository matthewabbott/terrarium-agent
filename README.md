# Terrarium Agent

A flexible agent runtime with sandboxed execution, vLLM integration, and extensible tool system.

## Architecture

**Agent-first design** where tools (IRC, code execution, web, files) are integrated into a unified agent runtime, rather than building tool-specific bots.

### Components

- **Agent Runtime**: Core loop with context management and tool orchestration
- **vLLM Integration**: GLM-Air-4.5 (4-bit quantized) via vLLM server
- **Tool System**: Pluggable tools with clean interfaces
- **Context Swapping**: Different personas/contexts per domain (IRC ambassador, coder, etc.)
- **Sandbox**: Safe execution environment for code and commands

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

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp config/agent.yaml.example config/agent.yaml
# Edit config/agent.yaml with your settings

# Run vLLM server (separate terminal)
vllm serve glm-air-4.5 --quantization awq --dtype half

# Run agent
python main.py
```

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

## Related Projects

- [terrarium-irc](../terrarium-irc) - IRC bot/client used by IRC tool