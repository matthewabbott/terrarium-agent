# Session Management for Clients

This guide explains how to manage conversation sessions when using the Terrarium Agent HTTP API.

## Overview

**Key Principle:** The agent server is stateless. **You** (the client) manage conversation history.

Every request to `/v1/chat/completions` includes the full conversation history:
- System prompt (personality/instructions)
- Previous user messages
- Previous assistant responses

The server processes the request and returns a new response. It doesn't remember anything after the request completes.

## Message Format

Follow OpenAI's chat completion format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful IRC bot. Be concise and friendly."
    },
    {
      "role": "user",
      "content": "alice: What is Python?"
    },
    {
      "role": "assistant",
      "content": "Python is a programming language known for readability and versatility."
    },
    {
      "role": "user",
      "content": "bob: Can you show an example?"
    }
  ]
}
```

### Roles

- **system**: Instructions for the agent (personality, constraints, context)
  - Appears first in the messages array
  - Sets the tone and behavior
  - Example: "You are a helpful IRC bot. Be concise."

- **user**: Messages from users
  - For IRC: Prefix with username (`alice: What is Python?`)
  - For web: Just the user's message
  - For games: Game state or user action

- **assistant**: Previous responses from the agent
  - Add these to maintain conversation flow
  - Agent uses them for context

## Client-Side Session Management

### Simple Pattern (In-Memory)

```python
class ConversationSession:
    """Manage a single conversation session in memory."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages = []

    def add_user_message(self, content: str):
        """Add user message to history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add assistant response to history."""
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self):
        """Get full conversation history with system prompt."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]

    def clear(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = []


# Usage
session = ConversationSession("You are a helpful assistant.")
session.add_user_message("What is 2+2?")

# Send to agent
response = agent_client.chat(session.get_messages())

# Add response to history
session.add_assistant_message(response)

# Next turn has context
session.add_user_message("What about 3+3?")
response = agent_client.chat(session.get_messages())  # Knows we're doing math
```

### Persistent Pattern (File/Database)

For IRC bots or web apps that need persistence across restarts:

```python
import json
from pathlib import Path
from datetime import datetime

class PersistentSession:
    """Manage a conversation session with file persistence."""

    def __init__(self, session_id: str, storage_dir: str = "./sessions"):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.session_file = self.storage_dir / f"{session_id}.json"
        self.system_prompt = "You are a helpful assistant."
        self.messages = []

        # Load existing session
        if self.session_file.exists():
            self.load()

    def add_user_message(self, content: str):
        """Add user message and save."""
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def add_assistant_message(self, content: str):
        """Add assistant response and save."""
        self.messages.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def get_messages(self):
        """Get conversation history without timestamps."""
        return [
            {"role": "system", "content": self.system_prompt},
            *[{"role": m["role"], "content": m["content"]} for m in self.messages]
        ]

    def save(self):
        """Save session to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "messages": self.messages
        }
        self.session_file.write_text(json.dumps(data, indent=2))

    def load(self):
        """Load session from disk."""
        data = json.loads(self.session_file.read_text())
        self.system_prompt = data.get("system_prompt", self.system_prompt)
        self.messages = data.get("messages", [])


# Usage
session = PersistentSession("main")
session.add_user_message("What is Python?")
response = agent_client.chat(session.get_messages())
session.add_assistant_message(response)
# Automatically saved to disk

# Later or after restart
session = PersistentSession("main")  # Loads previous conversation
session.add_user_message("Can you elaborate?")  # Continues where left off
```

## Context Window Management

### The Problem

LLMs have limited context windows (number of tokens they can process). For GLM-4.5-Air, this is typically 4096-8192 tokens.

A long conversation can exceed this limit.

### Solution: Sliding Window

Keep only recent messages:

```python
def get_messages(self, max_messages: int = 20):
    """Get recent conversation history with system prompt."""
    recent = self.messages[-max_messages:]  # Last N messages
    return [
        {"role": "system", "content": self.system_prompt},
        *recent
    ]
```

**Advantages:**
- Prevents context overflow
- Keeps recent/relevant information
- Conversation stays focused

**Trade-offs:**
- Loses older context
- Agent "forgets" earlier conversation

### Advanced: Summarization

For longer conversations, summarize old messages:

```python
def get_messages_with_summary(self):
    """Get messages with summarized history."""
    if len(self.messages) <= 10:
        # Short conversation - send everything
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]
    else:
        # Long conversation - summarize middle, keep recent
        old_messages = self.messages[:len(self.messages)-10]
        recent_messages = self.messages[-10:]

        # Generate summary (make a separate agent request)
        summary = generate_summary(old_messages)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Previous conversation summary: {summary}"},
            *recent_messages
        ]
```

## Best Practices

### System Prompts

Good system prompts are:
- **Specific**: "You are an IRC bot in #python. Help with Python questions. Be concise (2-3 sentences max)."
- **Contextual**: Include relevant constraints or domain knowledge
- **Consistent**: Use the same prompt for the same conversation

Bad system prompts:
- Too vague: "You are helpful."
- Too long: (1000+ words of instructions)
- Changing: Different prompt each request breaks context

### Session Organization

**For IRC bots:**
```python
# One session per channel
sessions = {
    "#python": PersistentSession("irc_python"),
    "#linux": PersistentSession("irc_linux"),
}
```

**For web apps:**
```python
# One session per user or per browser session
session_id = request.cookies.get("session_id")
session = PersistentSession(f"web_{session_id}")
```

**For games:**
```python
# One session per game instance
session = PersistentSession(f"game_{game_id}")
```

### Storage Location

**Development:**
```python
storage_dir = "./sessions"  # Relative to project
```

**Production:**
```python
storage_dir = "/var/lib/myapp/sessions"  # Absolute path
```

**Backup:**
Sessions are just JSON files - easy to backup:
```bash
tar -czf sessions-backup.tar.gz sessions/
```

### Message Cleanup

Periodically clean up old sessions:

```python
from datetime import datetime, timedelta

def cleanup_old_sessions(storage_dir, days=30):
    """Delete sessions older than N days."""
    cutoff = datetime.now() - timedelta(days=days)

    for session_file in Path(storage_dir).glob("*.json"):
        if session_file.stat().st_mtime < cutoff.timestamp():
            session_file.unlink()  # Delete
```

## Examples

### IRC Bot (Per-Channel Sessions)

```python
from client_library import AgentClient

client = AgentClient()
sessions = {}

def on_message(channel, user, message):
    """Handle IRC message with persistent context."""
    # Get or create session for this channel
    if channel not in sessions:
        sessions[channel] = PersistentSession(
            f"irc_{channel.lstrip('#')}",
            storage_dir="./irc_sessions"
        )
        sessions[channel].system_prompt = (
            f"You are a helpful IRC bot in {channel}. "
            "Be concise and friendly. Answer questions briefly."
        )

    session = sessions[channel]

    # Add user message (with username)
    session.add_user_message(f"{user}: {message}")

    # Get response from agent
    response = client.chat(session.get_messages())

    # Add response to history
    session.add_assistant_message(response)

    # Send to IRC
    send_to_irc(channel, response)
```

### Web Chat (User Sessions)

```python
from flask import Flask, request, session
from client_library import AgentClient

app = Flask(__name__)
client = AgentClient()

@app.route("/chat", methods=["POST"])
def chat():
    """Handle web chat message."""
    user_message = request.json["message"]

    # Get or create session for this user
    session_id = session.get("session_id")
    if not session_id:
        session_id = generate_unique_id()
        session["session_id"] = session_id

    conv_session = PersistentSession(
        f"web_{session_id}",
        storage_dir="./web_sessions"
    )

    # Add message and get response
    conv_session.add_user_message(user_message)
    response = client.chat(conv_session.get_messages())
    conv_session.add_assistant_message(response)

    return {"response": response}
```

### Game (Stateful Environment)

```python
class GameSession:
    """Manage game session with agent."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.session = PersistentSession(
            f"game_{game_id}",
            storage_dir="./game_sessions"
        )
        self.session.system_prompt = (
            "You are playing a text adventure game. "
            "Describe what you see and suggest actions."
        )
        self.client = AgentClient()

    def take_action(self, action: str):
        """Process player action and get agent response."""
        # Add action to history
        self.session.add_user_message(f"Player action: {action}")

        # Get agent's description/response
        response = self.client.chat(self.session.get_messages())

        # Add to history
        self.session.add_assistant_message(response)

        return response
```

## See Also

- [INTEGRATION.md](INTEGRATION.md) - Complete integration guide
- [AGENT_API.md](AGENT_API.md) - HTTP API specification
- [client_library.py](client_library.py) - Python client with ConversationContext helper
