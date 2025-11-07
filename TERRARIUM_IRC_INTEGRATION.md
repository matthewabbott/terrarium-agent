# Integrating with Terrarium-IRC

Quick guide for connecting terrarium-irc to terrarium-agent.

## Summary

Replace Ollama backend in terrarium-irc with terrarium-agent, enabling:
- On-demand agent invocation (not always running)
- Better model (GLM-4.5-Air vs Qwen2.5)
- Tool calling and harness access
- Shared context system

## Setup (Option 1: Library Integration)

### Step 1: Update terrarium-irc dependencies

Add to `terrarium-irc/requirements.txt`:
```
# Terrarium Agent integration
-e ../terrarium-agent
```

### Step 2: Create agent client wrapper

Create `terrarium-irc/llm/agent_wrapper.py`:

```python
"""Wrapper for terrarium-agent integration."""

import sys
from pathlib import Path

# Add terrarium-agent to path
AGENT_PATH = Path(__file__).parent.parent.parent / "terrarium-agent"
if str(AGENT_PATH) not in sys.path:
    sys.path.insert(0, str(AGENT_PATH))

# Import agent client
from llm.agent_client import AgentClient


class TerrariumAgentClient:
    """
    Agent client that mimics Ollama LLMClient interface.

    Drop-in replacement for terrarium-irc's Ollama client.
    """

    def __init__(
        self,
        model: str = "glm-air-4.5",
        api_url: str = "http://localhost:8000",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """Initialize with Ollama-compatible interface."""
        # Create agent client with vLLM backend
        self._agent_client = AgentClient(
            vllm_url=api_url,
            vllm_model=model
        )
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._initialized = False

    async def initialize(self):
        """Initialize agent."""
        if not self._initialized:
            await self._agent_client.initialize()
            self._initialized = True

    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        context: str = None
    ) -> str:
        """
        Generate response (Ollama-compatible interface).

        Args:
            prompt: User prompt
            system_prompt: System prompt (ignored, uses IRC context)
            context: IRC chat history

        Returns:
            Generated response
        """
        # Ensure initialized
        await self.initialize()

        # Use agent client
        response = await self._agent_client.generate(
            prompt=prompt,
            context=context,
            context_name="irc"
        )

        return response

    async def shutdown(self):
        """Shutdown agent."""
        if self._agent_client:
            await self._agent_client.shutdown()
            self._initialized = False
```

### Step 3: Update terrarium-irc main.py

Option A - Environment Variable (Recommended):

```python
# In terrarium-irc/main.py

import os
from llm import LLMClient  # Ollama

# Check backend selection
llm_backend = os.getenv('LLM_BACKEND', 'ollama')

if llm_backend == 'agent':
    print("Using terrarium-agent backend...")
    from llm.agent_wrapper import TerrariumAgentClient as LLMClient
    llm_api_url = os.getenv('LLM_API_URL', 'http://localhost:8000')  # vLLM URL
else:
    print("Using Ollama backend...")
    from llm import LLMClient
    llm_api_url = os.getenv('LLM_API_URL', 'http://localhost:11434')  # Ollama URL

# Initialize as before
llm_client = LLMClient(
    model=llm_model,
    api_url=llm_api_url,
    temperature=llm_temperature,
    max_tokens=llm_max_tokens
)
```

Then in `.env`:
```bash
# Use terrarium-agent backend
LLM_BACKEND=agent
LLM_API_URL=http://localhost:8000  # vLLM server
```

Option B - Direct Replacement (Simpler):

```python
# Replace this import:
# from llm import LLMClient

# With this:
from llm.agent_wrapper import TerrariumAgentClient as LLMClient

# Everything else stays the same!
```

## Usage

### Start Services

Terminal 1 - vLLM Server:
```bash
cd terrarium-agent
./start_vllm_docker.sh
```

Terminal 2 - IRC Bot:
```bash
cd terrarium-irc
source venv/bin/activate
export LLM_BACKEND=agent  # if using env var approach
python main.py
```

### IRC Commands (Same as Before!)

```irc
<user> .terrarium what's the weather?
<bot> user: Let me check the recent conversation...

<user> .ask what is 2+2?
<bot> user: 2+2 equals 4.
```

## How It Works

```
IRC User: ".terrarium what did alice say?"
    ↓
terrarium-irc: Fetch recent messages from database
    ↓
AgentClient: Create/reuse agent runtime
    ↓
AgentRuntime: Load IRC context, process query
    ↓
vLLM: Generate response with GLM-4.5-Air
    ↓
AgentRuntime: Return response
    ↓
terrarium-irc: Send to IRC channel
```

## Performance

- **First query**: ~2-5 seconds (agent initialization)
- **Subsequent queries**: ~1-3 seconds (inference only)
- **vLLM startup**: ~1-2 minutes (one-time, can stay running)

## Resource Management

### Keep Agent Warm (Recommended)
- Agent client initializes on first query
- Stays in memory for subsequent queries
- Minimal overhead (~100MB RAM)

### Per-Query Initialization
Add to agent wrapper if you want cleanup:
```python
async def generate(...):
    await self.initialize()
    response = await self._agent_client.generate(...)
    await self._agent_client.shutdown()  # Clean up after each query
    return response
```

## Benefits

✅ Better model (GLM-4.5-Air 12B active vs Qwen2.5:7b)
✅ Tool calling support (agent can use IRC tools if needed)
✅ Harness access (agent can play games!)
✅ Unified backend (one vLLM server for all projects)
✅ IRC bot stays lightweight
✅ Agent resources used only when needed

## Troubleshooting

### "Connection refused" error
- Check vLLM is running: `curl http://localhost:8000/health`
- Start vLLM: `cd terrarium-agent && ./start_vllm_docker.sh`

### Slow first response
- Normal! Agent initializes on first query (~2-5s)
- Subsequent queries are faster
- Keep vLLM server running to avoid model loading time

### IRC bot crashes
- Check vLLM is accessible
- Try Ollama fallback: `export LLM_BACKEND=ollama`
- Check logs for errors

### Model not loaded in vLLM
- Verify model download complete: `./check_model.sh`
- Check vLLM startup logs for errors
- Ensure GLM-4.5-Air-AWQ-4bit in models/ directory

## Testing

Test agent client directly (no IRC needed):

```bash
cd terrarium-agent
source venv/bin/activate
python llm/agent_client.py
```

Test from IRC bot:

```bash
cd terrarium-irc
source venv/bin/activate
export LLM_BACKEND=agent
python -c "
import asyncio
from llm.agent_wrapper import TerrariumAgentClient

async def test():
    client = TerrariumAgentClient()
    response = await client.generate('Hello!')
    print(f'Response: {response}')
    await client.shutdown()

asyncio.run(test())
"
```

## Future Enhancements

### 1. Invoke Harnesses via IRC
```
<user> .play number_guess
<bot> Starting game... I'm thinking of a number between 1 and 100!
```

### 2. Multi-Context Support
```
<user> .ask [research] explain quantum computing
<bot> [loads research context instead of IRC context]
```

### 3. Tool Discovery
```
<user> .tools
<bot> Available: IRC history search, web search, file operations, harnesses (number_guess, text_adventure)
```

## Migration Checklist

- [ ] Model download complete (`./check_model.sh`)
- [ ] vLLM server starts successfully
- [ ] AgentClient test passes
- [ ] terrarium-irc updated with agent wrapper
- [ ] LLM_BACKEND set to "agent" in .env
- [ ] Test `.ask` command in test channel
- [ ] Test `.terrarium` command with IRC context
- [ ] Monitor latency and quality
- [ ] Switch production channels

## Rollback

If issues occur, revert to Ollama:

```bash
# In terrarium-irc/.env
LLM_BACKEND=ollama
LLM_API_URL=http://localhost:11434

# Restart IRC bot
```

Original behavior restored immediately!
