# Integration Guide

This guide explains how to integrate external applications with the Terrarium Agent HTTP API.

## Overview

**Recommended Integration Method:** HTTP API (server.py on port 8080)

**Architecture:**
- **Stateless Server:** Server doesn't store conversation history
- **Client Responsibility:** Your application manages conversation contexts
- **OpenAI Compatible:** Uses standard OpenAI chat completion format
- **Streaming Support:** Optional SSE streaming for real-time responses

**Why HTTP API?**
- Process isolation (agent crashes don't affect your app)
- Language-agnostic (works with any HTTP client)
- Simple deployment (start server, make requests)
- Concurrent clients supported

## Quick Start (terrarium-irc)

### 1. Start the Services

```bash
# Terminal 1: Start vLLM (one-time setup)
cd terrarium-agent
./start_vllm_docker.sh
# Wait for: "Application startup complete"

# Terminal 2: Start agent HTTP API server
cd terrarium-agent
source venv/bin/activate
python server.py
# Server runs on http://localhost:8080
```

### 2. Make Requests from terrarium-irc

```python
import requests

def get_agent_response(channel_history: list[dict]) -> str:
    """
    Get agent response for IRC channel.

    Args:
        channel_history: Conversation history in OpenAI format
            [{"role": "system", "content": "You are a helpful IRC bot."},
             {"role": "user", "content": "alice: What is Python?"},
             {"role": "assistant", "content": "Python is..."},
             {"role": "user", "content": "bob: Can you show an example?"}]

    Returns:
        Agent's response text
    """
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "messages": channel_history,
            "temperature": 0.8,
            "max_tokens": 512
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
```

### 3. Manage Conversation History

```python
class IRCChannelContext:
    """Manage conversation history for an IRC channel."""

    def __init__(self, channel: str):
        self.channel = channel
        self.system_prompt = (
            "You are a helpful IRC bot assistant. "
            "Be concise and friendly. Avoid long responses."
        )
        self.messages = []

    def add_user_message(self, user: str, message: str):
        """Add IRC message to history."""
        self.messages.append({
            "role": "user",
            "content": f"{user}: {message}"
        })

    def add_assistant_response(self, response: str):
        """Add agent response to history."""
        self.messages.append({
            "role": "assistant",
            "content": response
        })

    def get_history(self, last_n: int = 10):
        """Get recent conversation history with system prompt."""
        recent = self.messages[-last_n:]  # Keep last N messages
        return [
            {"role": "system", "content": self.system_prompt},
            *recent
        ]

    def clear(self):
        """Clear conversation history."""
        self.messages = []


# Usage in IRC bot
channel_contexts = {}

def on_message(channel: str, user: str, message: str):
    """Handle IRC message."""
    # Get or create context for this channel
    if channel not in channel_contexts:
        channel_contexts[channel] = IRCChannelContext(channel)

    context = channel_contexts[channel]

    # Add user message
    context.add_user_message(user, message)

    # Get agent response
    try:
        response = get_agent_response(context.get_history())
        context.add_assistant_response(response)

        # Send to IRC channel
        send_to_irc(channel, response)

    except requests.RequestException as e:
        print(f"Agent error: {e}")
        # Optionally: Fallback to simple responses or error message
```

## Python Client Implementation

### Simple Client (No Dependencies Beyond Requests)

```python
import requests
from typing import List, Dict, Optional

class AgentClient:
    """Simple HTTP client for Terrarium Agent API."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate chat response.

        Args:
            messages: Conversation history (OpenAI format)
            temperature: Sampling temperature 0.0-2.0
            max_tokens: Maximum tokens to generate

        Returns:
            Assistant's response text

        Raises:
            requests.RequestException: If request fails
        """
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=60
        )

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        """Check if agent server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
```

### Using the Official Client Library

Terrarium Agent provides a more robust client with retry logic and streaming support:

```python
# Copy client_library.py to your project or install as package
from client_library import AgentClient, ConversationContext

# Create client
client = AgentClient("http://localhost:8080")

# Simple generation
response = client.generate("What is Python?")

# Chat with history
context = ConversationContext("You are a helpful IRC bot.")
context.add_user_message("alice: What is a decorator?")
response = client.chat(context.get_messages())
context.add_assistant_message(response)

# Streaming (for web UIs)
for chunk in client.chat_stream(context.get_messages()):
    print(chunk, end="", flush=True)
```

## Session Management Best Practices

### Message Format

Follow OpenAI's chat completion format:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful IRC bot. Be concise."
    },
    {
        "role": "user",
        "content": "alice: What is Python?"
    },
    {
        "role": "assistant",
        "content": "Python is a programming language..."
    },
    {
        "role": "user",
        "content": "bob: Can you show an example?"
    }
]
```

**Roles:**
- `system`: Instructions for the agent (personality, constraints)
- `user`: Messages from users (prefix with username for IRC)
- `assistant`: Previous agent responses

### Context Window Management

**Problem:** LLM context windows have limits (typically 4096-8192 tokens for GLM-4.5-Air)

**Solution:** Keep only recent messages

```python
def get_history(self, max_messages: int = 10):
    """Get recent conversation with system prompt."""
    recent = self.messages[-max_messages:]
    return [
        {"role": "system", "content": self.system_prompt},
        *recent
    ]
```

**Advanced:** Implement sliding window with summarization:
```python
# Keep: First message (system prompt) + last N messages + summary of middle
# This preserves context while staying under token limits
```

### Per-Channel Contexts

For IRC, maintain separate conversation histories per channel:

```python
channel_contexts = {
    "#python": IRCChannelContext("#python"),
    "#javascript": IRCChannelContext("#javascript"),
}
```

Each channel has its own conversation thread, preventing cross-talk.

## Error Handling

### Health Checks

Check agent availability before making requests:

```python
client = AgentClient()

if not client.health_check():
    print("Agent server not available")
    # Fallback: Use simple responses or notify admins
else:
    response = client.chat(messages)
```

### Retry Logic

Handle transient failures with exponential backoff:

```python
import time

def get_agent_response_with_retry(messages, max_retries=3):
    """Get response with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return client.chat(messages)
        except requests.Timeout:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)
            else:
                raise
        except requests.ConnectionError:
            # Server down - don't retry
            raise
```

### Graceful Degradation

Provide fallback behavior when agent is unavailable:

```python
def on_message(channel, user, message):
    try:
        response = get_agent_response(context.get_history())
        send_to_irc(channel, response)
    except requests.RequestException as e:
        # Log error
        print(f"Agent error: {e}")

        # Fallback: Simple keyword responses
        if "help" in message.lower():
            send_to_irc(channel, "Agent is temporarily unavailable. Try /help for commands.")
```

## Examples

### curl (Testing)

```bash
# Non-streaming request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50
  }'

# Streaming request
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count to 5"}
    ],
    "stream": true
  }'
```

### JavaScript/Node.js

```javascript
// Using fetch API
async function getAgentResponse(messages) {
  const response = await fetch('http://localhost:8080/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: messages,
      temperature: 0.7,
      max_tokens: 512
    })
  });

  const data = await response.json();
  return data.choices[0].message.content;
}

// Usage
const messages = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is Node.js?' }
];

const response = await getAgentResponse(messages);
console.log(response);
```

## API Reference

For complete API specification, see [AGENT_API.md](AGENT_API.md).

**Key Endpoints:**
- `POST /v1/chat/completions` - Main chat endpoint (OpenAI-compatible)
- `POST /v1/generate` - Simple single-turn generation
- `GET /health` - Health check
- `GET /v1/models` - List available models

**Parameters:**
- `messages` - Conversation history (required)
- `temperature` - Sampling temperature 0.0-2.0 (optional, default: 0.7)
- `max_tokens` - Max tokens to generate (optional, default: 2048)
- `stream` - Enable streaming (optional, default: false)

## Troubleshooting

### "Connection refused" Error

**Cause:** Agent server not running

**Solution:**
```bash
# Check if server is running
curl http://localhost:8080/health

# If not, start it
python server.py
```

### "vLLM server not ready" Error

**Cause:** vLLM container not running or still loading model

**Solution:**
```bash
# Check vLLM status
docker ps | grep vllm
docker logs vllm-server

# Start if not running
./start_vllm_docker.sh
```

### Slow Responses

**Cause:** Model generation is compute-intensive

**Expected:** 5-30 seconds for typical responses

**Solutions:**
- Use streaming (`stream: true`) for real-time feedback
- Reduce `max_tokens` for faster responses
- Set appropriate client timeouts (60-120 seconds)

### Context Too Long Error

**Cause:** Conversation history exceeds model's context window

**Solution:** Implement sliding window (keep only recent messages)
```python
recent_messages = messages[-10:]  # Keep last 10 messages
```

## Migration from Other LLM Services

### From Ollama

**Similarities:**
- Both use HTTP APIs
- Similar request/response format

**Differences:**
- Terrarium Agent uses OpenAI format (not Ollama's custom format)
- Stateless (you manage conversation history)

**Migration:**
```python
# Ollama
response = ollama.chat(model='llama2', messages=messages)

# Terrarium Agent
response = requests.post(
    'http://localhost:8080/v1/chat/completions',
    json={'messages': messages}
).json()['choices'][0]['message']['content']
```

### From OpenAI API

**Good news:** Terrarium Agent is OpenAI-compatible!

**Change:**
```python
# OpenAI
from openai import OpenAI
client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)

# Terrarium Agent (local)
import requests
response = requests.post(
    'http://localhost:8080/v1/chat/completions',
    json={'messages': messages}
).json()
```

Or use OpenAI's client library with Terrarium Agent:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # No auth required for local server
)

response = client.chat.completions.create(
    model="glm-4.5-air",
    messages=messages
)
```

## Next Steps

1. **Copy client implementation** to your project (simple client or client_library.py)
2. **Test connection** with health check
3. **Implement conversation management** for your use case (IRC channels, web sessions, etc.)
4. **Add error handling** and retry logic
5. **Deploy** with proper monitoring and fallback strategies

For more details:
- [AGENT_API.md](AGENT_API.md) - Complete API specification
- [QUICKSTART.md](QUICKSTART.md) - Server setup and configuration
- [client_library.py](client_library.py) - Full-featured Python client with examples
