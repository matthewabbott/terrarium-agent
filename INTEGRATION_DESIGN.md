# Terrarium-IRC Integration Design

## Overview

Integrate terrarium-agent with terrarium-irc to enable on-demand agent invocation from IRC.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  terrarium-irc (Always Running)                          │
│  ├─ IRC client (connects to channels)                    │
│  ├─ Message logger (SQLite)                              │
│  ├─ Command handler (.terrarium, .ask, etc.)             │
│  └─ AgentClient (NEW) ────────────────────┐              │
└────────────────────────────────────────────│──────────────┘
                                             │
                                             │ invoke
                                             ↓
┌──────────────────────────────────────────────────────────┐
│  terrarium-agent (On-Demand)                             │
│  ├─ AgentRuntime                                         │
│  ├─ Context Manager (IRC context)                        │
│  ├─ Tools (IRC tool, harnesses, etc.)                    │
│  └─ vLLM Client ───────────────────┐                     │
└────────────────────────────────────│──────────────────────┘
                                     │
                                     │ query
                                     ↓
┌──────────────────────────────────────────────────────────┐
│  vLLM Server (Warm/Always On)                            │
│  └─ GLM-4.5-Air-AWQ-4bit                                 │
└──────────────────────────────────────────────────────────┘
```

## Integration Options

### Option 1: Library Integration (Recommended for v1)

**Implementation:**
- terrarium-irc imports terrarium-agent as a Python library
- Replace Ollama `LLMClient` with `AgentClient`
- AgentRuntime created on-demand per query

**Pros:**
- ✅ Simplest to implement
- ✅ No network overhead
- ✅ Shared venv/dependencies
- ✅ Easy debugging

**Cons:**
- ⚠️ Same process (crash affects both)
- ⚠️ Shared resources

**Code Pattern:**
```python
# In terrarium-irc/bot/commands.py

from terrarium_agent.agent.runtime import AgentRuntime
from terrarium_agent.llm.vllm_client import VLLMClient

class AgentClient:
    def __init__(self, vllm_url="http://localhost:8000"):
        self.vllm_url = vllm_url
        self.runtime = None

    async def initialize(self):
        """Initialize agent runtime (on-demand)."""
        if not self.runtime:
            llm = VLLMClient(base_url=self.vllm_url)
            self.runtime = AgentRuntime(llm_client=llm)
            await self.runtime.initialize()

    async def generate(self, prompt, context=None, system_prompt=None):
        """Generate response using agent."""
        await self.initialize()

        # Build context-aware prompt
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"

        # Use IRC context
        response = await self.runtime.process(
            prompt=full_prompt,
            context_name="irc"
        )
        return response
```

### Option 2: HTTP API (Future Enhancement)

**Implementation:**
- terrarium-agent runs FastAPI server
- terrarium-irc calls agent via HTTP POST
- Separate processes, clean isolation

**Pros:**
- ✅ Process isolation (more robust)
- ✅ Independent restart/upgrade
- ✅ Can run on different machines
- ✅ Multiple clients can use agent

**Cons:**
- ⚠️ Network latency (minimal)
- ⚠️ Extra service to manage
- ⚠️ Authentication/security considerations

**API Design:**
```python
# terrarium-agent/api.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    prompt: str
    context: str | None = None
    context_name: str = "irc"

class QueryResponse(BaseModel):
    response: str
    metadata: dict = {}

@app.post("/query")
async def query_agent(request: QueryRequest) -> QueryResponse:
    # Invoke agent runtime
    response = await agent_runtime.process(
        prompt=request.prompt,
        context_name=request.context_name,
        additional_context=request.context
    )
    return QueryResponse(response=response)
```

## Implementation Plan (Option 1)

### Phase 1: Create AgentClient in terrarium-agent

**File: `terrarium-agent/llm/agent_client.py`**

```python
"""Agent client for external invocation (like from terrarium-irc)."""

from typing import Optional
from pathlib import Path

from .vllm_client import VLLMClient
from agent.runtime import AgentRuntime
from agent.context import ContextManager


class AgentClient:
    """
    Client interface for invoking terrarium-agent from external tools.

    Designed for on-demand agent invocation with minimal overhead.
    Handles agent lifecycle: initialize → process → (optional) shutdown.
    """

    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        vllm_model: str = "glm-air-4.5",
        contexts_dir: Optional[Path] = None
    ):
        """
        Initialize agent client.

        Args:
            vllm_url: vLLM server URL
            vllm_model: Model name
            contexts_dir: Path to context definitions
        """
        self.vllm_url = vllm_url
        self.vllm_model = vllm_model
        self.contexts_dir = contexts_dir or Path(__file__).parent.parent / "config" / "contexts"

        # Runtime is created on-demand
        self._runtime = None
        self._initialized = False

    async def initialize(self):
        """Initialize agent runtime (lazy initialization)."""
        if self._initialized:
            return

        # Create vLLM client
        llm = VLLMClient(
            base_url=self.vllm_url,
            model=self.vllm_model
        )

        # Create context manager
        context_manager = ContextManager(contexts_dir=self.contexts_dir)

        # Create runtime
        self._runtime = AgentRuntime(
            llm_client=llm,
            context_manager=context_manager
        )

        # Initialize
        await self._runtime.initialize()
        self._initialized = True

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        context_name: str = "irc"
    ) -> str:
        """
        Generate response using agent.

        Args:
            prompt: User prompt/question
            system_prompt: Override system prompt (optional)
            context: Additional context (e.g., IRC history)
            context_name: Context to load (default: irc)

        Returns:
            Generated response
        """
        await self.initialize()

        # Build full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"

        # Process with agent
        response = await self._runtime.process(
            prompt=full_prompt,
            context_name=context_name
        )

        return response

    async def shutdown(self):
        """Shutdown agent runtime."""
        if self._runtime:
            await self._runtime.shutdown()
            self._runtime = None
            self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
```

### Phase 2: Update terrarium-irc

**File: `terrarium-irc/llm/agent_client.py`** (new)

```python
"""Client for terrarium-agent integration."""

import sys
from pathlib import Path

# Add terrarium-agent to path
AGENT_PATH = Path(__file__).parent.parent.parent / "terrarium-agent"
sys.path.insert(0, str(AGENT_PATH))

from llm.agent_client import AgentClient as TerrariumAgentClient

# Re-export with IRC-specific defaults
class AgentClient(TerrariumAgentClient):
    """IRC-specific agent client wrapper."""

    def __init__(self, vllm_url="http://localhost:8000"):
        super().__init__(
            vllm_url=vllm_url,
            vllm_model="glm-air-4.5",
            contexts_dir=AGENT_PATH / "config" / "contexts"
        )
```

**Update: `terrarium-irc/main.py`**

```python
# Change this:
from llm import LLMClient

# To this:
from llm.agent_client import AgentClient as LLMClient
# Note: Same interface, drop-in replacement!
```

### Phase 3: IRC Context Configuration

**File: `terrarium-agent/config/contexts/irc.yaml`** (already exists!)

Just ensure it's properly configured:

```yaml
name: irc
description: IRC channel ambassador - conversational and helpful
system_prompt: |
  You are a helpful IRC bot called Terrarium. You assist users in IRC channels
  by answering questions and participating in conversations.

  Be concise (IRC has message limits), friendly, and helpful.
  You have access to recent channel history for context.

tools:
  - irc  # IRC tool (if needed for additional IRC operations)

personality:
  - Conversational and friendly
  - Concise responses (IRC-appropriate)
  - Uses channel context effectively
```

## Deployment Workflow

### Standard Operation

1. **Start vLLM server** (stays running):
   ```bash
   cd terrarium-agent
   ./start_vllm_docker.sh
   ```

2. **Start terrarium-irc** (always on):
   ```bash
   cd terrarium-irc
   source venv/bin/activate
   python main.py
   ```

3. **On IRC command** (`.terrarium <question>`):
   - IRC bot fetches recent context from database
   - Creates AgentClient (or reuses if warm)
   - Calls `agent_client.generate(prompt, context=irc_history)`
   - Agent runtime loads IRC context
   - vLLM generates response
   - IRC bot sends response to channel

### Resource Management Options

**Option A: Lazy Initialization (Recommended)**
- Agent runtime created on first query
- Stays warm for subsequent queries
- Shuts down on IRC bot restart

**Option B: Keep Warm**
- Initialize agent on IRC bot startup
- Always ready (faster first response)
- Uses more resources

**Option C: Per-Query**
- Create new agent runtime per query
- Shutdown after response
- Minimal resource usage, slower

## Integration Benefits

### For IRC Users
- ✅ Same commands (`.terrarium`, `.ask`)
- ✅ Better responses (GLM-4.5-Air vs Qwen2.5:7b)
- ✅ Tool calling support (agent can use IRC history tools)
- ✅ Context management (IRC context loaded automatically)

### For Development
- ✅ Single LLM backend (vLLM) for all terrarium projects
- ✅ Harnesses available to agent (can play games via IRC!)
- ✅ Shared context system
- ✅ Easier to add new capabilities

### For Operations
- ✅ IRC bot is lightweight (always responsive)
- ✅ Agent runtime on-demand (resource efficient)
- ✅ vLLM server stays warm (fast inference)
- ✅ Can restart components independently

## Migration Path

### Step 1: Parallel Operation
- Keep Ollama backend as fallback
- Add agent backend as optional
- Use env var to switch: `LLM_BACKEND=agent` vs `LLM_BACKEND=ollama`

### Step 2: Test Period
- Use agent backend in test channel
- Monitor performance, latency, quality
- Fix any issues

### Step 3: Full Migration
- Switch all channels to agent backend
- Remove Ollama dependency
- Optionally remove Ollama LLMClient code

## Testing Plan

### Unit Tests
```python
# test_agent_integration.py

async def test_agent_client_generation():
    """Test agent client generates responses."""
    client = AgentClient(vllm_url="http://localhost:8000")

    response = await client.generate(
        prompt="What is 2+2?",
        context="Previous messages: [test context]"
    )

    assert response
    assert len(response) > 0

    await client.shutdown()

async def test_agent_client_context_loading():
    """Test agent loads IRC context."""
    client = AgentClient()

    response = await client.generate(
        prompt="What did Alice say?",
        context="<alice> I like pizza\n<bob> Me too",
        context_name="irc"
    )

    assert "pizza" in response.lower()

    await client.shutdown()
```

### Integration Tests
1. Start vLLM server with GLM-4.5-Air
2. Run IRC bot with agent backend
3. Send `.terrarium` command in test channel
4. Verify response uses IRC context
5. Check latency is acceptable (<5s)

## Future Enhancements

### 1. Harness Invocation via IRC
```
User: .play number_guess
Bot: Starting NumberGuess harness...
Bot: Welcome to Number Guess! I'm thinking of a number...
[Agent plays game autonomously or with user input]
```

### 2. Agent Sessions
- Persistent conversations beyond single queries
- Agent can maintain state across IRC commands
- `/session start` and `/session end` commands

### 3. Multi-Agent Support
- Different agents for different purposes
- `.research <query>` uses research agent
- `.code <task>` uses coding agent

### 4. Tool Discovery
- Agent can search IRC history using database tools
- Agent can invoke harnesses
- Agent can access web, files, etc.

## Questions?

- **Q: Does vLLM need to always run?**
  A: For best performance, yes. But you can start it on-demand too.

- **Q: Can agent runtime stay warm between queries?**
  A: Yes! That's the default with Option A (lazy init).

- **Q: What about harnesses?**
  A: Harnesses are separate - agent can invoke them, but IRC bot doesn't need to know about them.

- **Q: Performance impact?**
  A: First query: ~2-5s (agent init). Subsequent: ~1-2s (inference only).

- **Q: Can I keep Ollama as backup?**
  A: Yes! Use env var to switch backends, or fall back if vLLM unavailable.
