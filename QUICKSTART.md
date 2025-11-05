# Quick Start Guide

## Prerequisites

1. **Python 3.10+** installed
2. **vLLM server** running with GLM-Air-4.5
3. **terrarium-irc** (optional, for IRC tool integration)

## Setup

### 1. Install Dependencies

```bash
cd terrarium-agent
pip install -r requirements.txt
```

### 2. Configure Agent

```bash
# Copy example config
cp config/agent.yaml.example config/agent.yaml

# Edit config with your settings
nano config/agent.yaml
```

### 3. Start vLLM Server

In a separate terminal:

```bash
# Install vLLM (if not already installed)
pip install vllm

# Download GLM-Air-4.5 model (if needed)
# Model should be in HuggingFace format, 4-bit quantized with AWQ/GPTQ

# Start vLLM server
vllm serve glm-air-4.5 \
  --quantization awq \
  --dtype half \
  --port 8000
```

Wait for vLLM to load the model (may take a minute).

## Running the Agent

### Interactive Mode

```bash
python main.py
```

You'll get an interactive prompt where you can:
- Chat with the agent
- Switch contexts with `/context <name>`
- List contexts with `/list`
- View tools with `/tools`

### Example Session

```
You: Hello!
Agent: Hello! I'm Terrarium, your assistant. How can I help you today?

You: /list
Available contexts:
  - irc *
  - coder

You: /context coder
Switched to context: coder

You: Write a function to calculate fibonacci
Agent: [Agent writes fibonacci function]

You: quit
```

## Contexts

### IRC Context

For interacting with IRC channels. Persona:
- Conversational and friendly
- Concise responses (IRC-appropriate)
- Has access to IRC tool

### Coder Context

For software development tasks. Persona:
- Technical and precise
- Detailed explanations
- Has access to code execution tools

## Tools

### IRC Tool (Mock)

Currently returns mock data. To integrate with real IRC:

1. Ensure `terrarium-irc` is installed
2. Update `tools/irc.py` to use actual IRC client
3. Configure IRC settings in `config/agent.yaml`

### Future Tools

- Python code execution
- Shell commands
- File operations
- Web search

## Development

### Project Structure

```
terrarium-agent/
├── agent/              # Core runtime and context management
│   ├── runtime.py      # Main agent loop
│   └── context.py      # Context swapping
├── llm/                # LLM integration
│   └── vllm_client.py  # vLLM API client
├── tools/              # Tool implementations
│   ├── base.py         # Base tool interface
│   └── irc.py          # IRC tool (mock)
├── config/             # Configuration
│   ├── contexts/       # Context YAML files
│   └── agent.yaml      # Main config
└── main.py             # Entry point
```

### Adding New Tools

1. Create new file in `tools/`
2. Extend `BaseTool` class
3. Implement `execute()` and `get_capabilities()`
4. Register in `main.py`

Example:

```python
from tools.base import BaseTool, ToolResult

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(name="mytool", description="Does something")

    async def execute(self, action: str, **kwargs) -> ToolResult:
        # Implementation
        pass

    def get_capabilities(self) -> List[Dict[str, Any]]:
        return [{"action": "do_thing", "description": "..."}]
```

### Adding New Contexts

1. Create YAML file in `config/contexts/`
2. Define system prompt, tools, personality
3. Context auto-loads on startup

See `config/contexts/irc.yaml` for example.

## Troubleshooting

### vLLM Not Responding

```bash
# Check if vLLM is running
curl http://localhost:8000/health

# Check vLLM logs for errors
```

### Model Not Found

Ensure GLM-Air-4.5 is:
- Downloaded and in correct format
- Quantized with AWQ or GPTQ
- Path correctly specified to vLLM

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Next Steps

1. **Implement real IRC tool** - Connect to actual terrarium-irc
2. **Add code execution tool** - Python sandbox
3. **Add shell tool** - Safe command execution
4. **Test with GLM-Air-4.5** - Verify model performance
5. **Implement tool calling** - Use `process_with_tools()` method

## Related Projects

- [terrarium-irc](../terrarium-irc) - IRC bot/client
- [vLLM](https://github.com/vllm-project/vllm) - LLM inference engine
