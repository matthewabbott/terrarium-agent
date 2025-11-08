# Terrarium Agent HTTP API

This document defines the HTTP API contract for Terrarium Agent server. External clients (terrarium-irc, web apps, game harnesses) use this API to get LLM responses.

## Architecture

```
┌─────────────────────────────────────┐
│ Terrarium Agent Server              │
│ - Always running (persistent)       │
│ - HTTP API on port 8080             │
│ - Request queue (FIFO)              │
│ - vLLM connection (warm)            │
│ - Stateless (no context storage)   │
└─────────────────────────────────────┘
          ↑           ↑           ↑
          │ HTTP      │ HTTP      │ HTTP
    ┌─────┴─────┐ ┌──┴────┐ ┌───┴────┐
    │ IRC Bot   │ │ Web   │ │ Game   │
    │           │ │ App   │ │Harness │
    │ Stores    │ │Stores │ │Stores  │
    │ context   │ │context│ │context │
    └───────────┘ └───────┘ └────────┘
```

**Key Design Principles:**
- **Stateless Server:** Server does not store conversation history
- **Client Responsibility:** Clients manage their own conversation contexts
- **Simple Queue:** FIFO request processing (one at a time, sequential)
- **OpenAI Compatible:** Uses OpenAI-compatible format where possible

## Base URL

```
http://localhost:8080
```

## Endpoints

### 1. Chat Completion (OpenAI-Compatible)

Generate a chat response given conversation history.

**Endpoint:** `POST /v1/chat/completions`

**Request:**
```json
{
  "model": "glm-4.5-air",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful IRC bot. Be concise."
    },
    {
      "role": "user",
      "content": "What's a Python decorator?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Request Fields:**
- `model` (string, optional): Model name (auto-detected if omitted)
- `messages` (array, required): Conversation history
  - Each message has `role` ("system", "user", "assistant") and `content`
- `temperature` (float, optional): Sampling temperature 0.0-2.0 (default: 0.7)
- `max_tokens` (integer, optional): Max tokens to generate (default: 2048)
- `stop` (array of strings, optional): Stop sequences
- `stream` (boolean, optional): Enable streaming (Server-Sent Events) (default: false)

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "/models/GLM-4.5-Air-AWQ-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A decorator in Python is a function that wraps another function..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 100,
    "total_tokens": 125
  }
}
```

**Errors:**
```json
{
  "error": {
    "message": "Invalid request: missing 'messages' field",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

**Status Codes:**
- 200: Success
- 400: Invalid request (missing fields, invalid format)
- 500: Server error (vLLM failure, etc.)
- 503: Server unavailable (vLLM not ready)

**Streaming Support:**

Enable streaming by setting `stream: true` in the request. Response will be Server-Sent Events (SSE) format:

```json
{
  "model": "glm-4.5-air",
  "messages": [{"role": "user", "content": "Count to 5"}],
  "stream": true
}
```

**Streaming Response** (SSE format):
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"glm-4.5-air","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"glm-4.5-air","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1699000000,"model":"glm-4.5-air","choices":[{"index":0,"delta":{"content":"2"},"finish_reason":null}]}

data: [DONE]
```

**SSE Format:**
- Content-Type: `text/event-stream`
- Each chunk prefixed with `data: `
- Stream ends with `data: [DONE]`
- Chunks have `delta` field (not `message`)

**Use Cases:**
- ✅ **Streaming**: Interactive web UIs, typing indicators, real-time feedback
- ✅ **Non-streaming**: IRC bots, batch jobs, simple integrations (default)

---

### 2. Simple Generate (Convenience Endpoint)

Simpler endpoint for single-turn generation (no conversation history).

**Endpoint:** `POST /v1/generate`

**Request:**
```json
{
  "prompt": "What is the capital of France?",
  "system_prompt": "You are a helpful assistant.",
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Request Fields:**
- `prompt` (string, required): User's message
- `system_prompt` (string, optional): System prompt
- `temperature` (float, optional): Sampling temperature (default: 0.7)
- `max_tokens` (integer, optional): Max tokens (default: 512)

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "model": "/models/GLM-4.5-Air-AWQ-4bit"
}
```

---

### 3. Health Check

Check if server is running and vLLM is ready.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "vllm_status": "ready",
  "model": "/models/GLM-4.5-Air-AWQ-4bit",
  "queue_length": 0
}
```

**Status Codes:**
- 200: Healthy and ready
- 503: Server starting or vLLM not ready

---

### 4. List Models

Get available models.

**Endpoint:** `GET /v1/models`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "/models/GLM-4.5-Air-AWQ-4bit",
      "object": "model",
      "created": 1699000000,
      "owned_by": "vllm"
    }
  ]
}
```

---

## Client Implementation Guide

### Python Example

```python
import requests

class AgentClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url

    def chat(self, messages, temperature=0.7, max_tokens=2048):
        """Send chat request with conversation history."""
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
        return response.json()["choices"][0]["message"]["content"]

    def generate(self, prompt, system_prompt=None):
        """Simple single-turn generation."""
        response = requests.post(
            f"{self.base_url}/v1/generate",
            json={
                "prompt": prompt,
                "system_prompt": system_prompt
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
```

### Usage in terrarium-irc

```python
# Initialize client
client = AgentClient("http://localhost:8080")

# Store conversation history per IRC channel
channel_histories = {}

def on_message(channel, user, message):
    # Get or create history for this channel
    if channel not in channel_histories:
        channel_histories[channel] = [
            {"role": "system", "content": "You are a helpful IRC bot. Be concise."}
        ]

    # Add user message
    channel_histories[channel].append({
        "role": "user",
        "content": f"{user}: {message}"
    })

    # Get response from agent
    try:
        response = client.chat(
            messages=channel_histories[channel],
            temperature=0.8,
            max_tokens=512
        )

        # Add assistant response to history
        channel_histories[channel].append({
            "role": "assistant",
            "content": response
        })

        # Send to IRC
        send_to_channel(channel, response)

    except requests.exceptions.RequestException as e:
        print(f"Agent error: {e}")
        send_to_channel(channel, "Agent temporarily unavailable")
```

---

## Server Behavior

### Request Queue

- **FIFO (First In, First Out):** Requests processed sequentially
- **No prioritization:** All requests equal priority (for now)
- **No concurrency:** One request at a time (leverages vLLM prefix caching)
- **Queue visibility:** `/health` endpoint shows queue length

### Context Management

- **Client Responsibility:** Server does NOT store conversation history
- **Stateless:** Each request is independent
- **Benefits:**
  - Simpler server implementation
  - Clients control their own data
  - No session management complexity
  - Easy to scale/restart server

### vLLM Integration

- **Persistent Connection:** Server maintains warm vLLM connection
- **Automatic Prefix Caching:** vLLM caches repeated message prefixes automatically
- **Model Auto-detection:** Server uses first available model if not specified
- **Error Handling:** Reconnects to vLLM if connection lost

---

## Performance Characteristics

### Latency

- **First request:** ~2-5s (cold start, model loading)
- **Subsequent requests:** ~1-3s (typical)
- **With prefix caching:** <1s (when conversation history reused)
- **Queue wait time:** Depends on queue length

### Throughput

- **Sequential processing:** One request at a time
- **Typical:** ~20-60 requests/minute (depends on response length)
- **vLLM optimization:** Prefix caching improves throughput for repeated contexts

### Optimization Tips

1. **Reuse conversations:** Send full history to leverage prefix caching
2. **Batch requests:** If possible, combine multiple questions
3. **Shorter prompts:** Reduce max_tokens for faster responses
4. **System prompt:** Keep system prompts consistent for better caching

---

## Error Handling

### Client-Side Best Practices

```python
import time
import requests

def call_agent_with_retry(client, messages, max_retries=3):
    """Call agent with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return client.chat(messages)
        except requests.exceptions.Timeout:
            print(f"Timeout, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.ConnectionError:
            print("Agent server not available")
            raise
        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                # Server error, retry
                time.sleep(2 ** attempt)
            else:
                # Client error, don't retry
                raise
    raise Exception("Max retries exceeded")
```

---

## Security Considerations

### Current Implementation

- **No authentication:** Localhost-only access
- **No rate limiting:** Trust clients to be reasonable
- **No input validation:** Basic validation only

### Future Enhancements

- API keys for authentication
- Rate limiting per client
- Input sanitization
- HTTPS/TLS support

---

## Running the Server

```bash
# Start vLLM (if not already running)
./start_vllm_docker.sh

# Start agent server
python server.py

# Server will start on http://localhost:8080
```

**Configuration:** Edit `server.py` or use environment variables:
```bash
export AGENT_PORT=8080
export VLLM_URL=http://localhost:8000
python server.py
```

---

## Testing

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Simple generation
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "system_prompt": "You are a helpful assistant."
  }'

# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

---

## See Also

- `server.py` - Reference server implementation
- `client_library.py` - Python client library
- `SESSION_STORAGE.md` - For client-side session management
- `CLAUDE.md` - Project overview
