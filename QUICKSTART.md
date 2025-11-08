# Quick Start Guide

## Prerequisites

1. **Docker** + **nvidia-container-toolkit** installed
2. **NVIDIA GPU:** GB10 (Blackwell) or compatible with driver 580+
3. **Python 3.12+** installed
4. **Model downloaded:** GLM-4.5-Air-AWQ-4bit in `models/` directory
5. **terrarium-irc** (optional, for IRC tool integration)

## Setup

### 1. Verify Prerequisites

```bash
# Check GPU and driver
nvidia-smi

# Check Docker
docker --version

# Check nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Check model
cd terrarium-agent
./check_model.sh
```

All should succeed before continuing.

### 2. Install Python Client Dependencies

```bash
# Create/activate venv
python3 -m venv venv
source venv/bin/activate

# Install lightweight client dependencies
pip install -r requirements.txt
```

**Note:** vLLM and PyTorch run in Docker, not in venv.

### 3. Start vLLM Server (Docker)

In a separate terminal:

```bash
cd terrarium-agent
./start_vllm_docker.sh
```

Wait for model to load (~2 minutes). Check logs:
```bash
docker logs -f vllm-server
# Look for: "Application startup complete"
```

## Running the Agent

### Option 1: HTTP API Server (Recommended for External Integration)

Start the agent server on port 8080:

```bash
source venv/bin/activate
python server.py
```

The server provides an OpenAI-compatible HTTP API that external applications (IRC bots, web apps, games) can call. See `AGENT_API.md` for full API documentation.

**Test the server:**
```bash
# Health check
curl http://localhost:8080/health

# Simple generation
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?"}'
```

**Use the Python client:**
```python
from client_library import AgentClient

client = AgentClient("http://localhost:8080")
response = client.generate("What is Python?")
print(response)
```

### Option 2: Simple Chat Interface

Interactive CLI for chatting with the agent:

```bash
source venv/bin/activate
python chat.py
```

Supports persistent sessions and conversation history. Type `/help` for commands.

### Option 3: Full Agent Runtime (with Tools & Harnesses)

Interactive prompt with full tool access:

```bash
source venv/bin/activate
python main.py
```

Features:
- Chat with the agent
- Switch contexts with `/context <name>`
- List contexts with `/list`
- View tools with `/tools`
- Start harness sessions with `/harness <name>`

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

## For External Integration (IRC, Web, Games)

If you're integrating Terrarium Agent into another application (like terrarium-irc), use the HTTP API approach:

**1. Start the Services:**
```bash
# Terminal 1: vLLM
./start_vllm_docker.sh

# Terminal 2: Agent server
python server.py
```

**2. Integrate from Your App:**
```python
# In terrarium-irc or your application
import requests

def get_agent_response(conversation_history):
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={"messages": conversation_history}
    )
    return response.json()["choices"][0]["message"]["content"]
```

**Complete Integration Guide:** See [INTEGRATION.md](INTEGRATION.md) for detailed examples including:
- IRC bot with per-channel sessions
- Conversation history management
- Error handling & retries
- Python client library usage

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
