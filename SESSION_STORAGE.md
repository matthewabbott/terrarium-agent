# Session Storage Format

This document describes the session storage system used by Terrarium Agent for managing persistent conversation contexts across multiple endpoints (IRC, games, web, CLI).

## Storage Structure

Sessions are organized by context type and date:

```
sessions/
├── cli/
│   ├── 2025-11-06/
│   │   ├── xyz123.json
│   │   └── main.json
│   └── 2025-11-05/
│       └── morning_chat.json
├── irc/
│   └── 2025-11-06/
│       ├── #python.json
│       └── #linux.json
├── game/
│   └── 2025-11-06/
│       ├── pokemon_run1.json
│       └── chess_game2.json
└── web/
    └── 2025-11-06/
        └── session_abc123.json
```

### Path Format

```
sessions/{context_type}/{date}/{session_id}.json
```

- **context_type**: Type of endpoint (cli, irc, game, web)
- **date**: YYYY-MM-DD format from session creation
- **session_id**: Unique identifier within context type

## Context ID Format

Full context identifier: `{type}:{id}`

Examples:
- `cli:main`
- `irc:#python`
- `game:pokemon_run1`
- `web:session_abc`

## Session File Format

Each session is stored as a JSON file with the following structure:

```json
{
  "session_id": "main",
  "context_type": "cli",
  "system_prompt": "You are a helpful AI assistant.",
  "messages": [
    {
      "role": "user",
      "content": "Hello!",
      "timestamp": "2025-11-06T14:30:00.123456"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you?",
      "timestamp": "2025-11-06T14:30:03.456789"
    }
  ],
  "metadata": {
    "created_at": "2025-11-06T14:30:00.123456",
    "last_active": "2025-11-06T14:35:12.987654",
    "message_count": 10,
    "session_id": "main",
    "context_type": "cli",
    "session_date": "2025-11-06"
  }
}
```

### Fields

**Top Level:**
- `session_id` (string): Unique identifier
- `context_type` (string): Context type (cli, irc, game, web)
- `system_prompt` (string): System prompt for the conversation
- `messages` (array): Conversation history
- `metadata` (object): Session metadata

**Message Object:**
- `role` (string): "user" or "assistant"
- `content` (string): Message text
- `timestamp` (string): ISO 8601 timestamp

**Metadata Object:**
- `created_at` (string): Session creation timestamp
- `last_active` (string): Last activity timestamp
- `message_count` (integer): Number of messages
- `session_id` (string): Session identifier
- `context_type` (string): Context type
- `session_date` (string): Date in YYYY-MM-DD format

## API Usage

### Python API

```python
from agent.multi_context_manager import MultiContextManager
from llm.vllm_client import VLLMClient

# Initialize
client = VLLMClient(base_url="http://localhost:8000")
manager = MultiContextManager(client, storage_dir="./sessions")

# Process with context (auto-saves)
response = await manager.process_with_context(
    context_id="irc:#python",
    user_message="What's a decorator?",
    system_prompt="You are a helpful IRC bot. Be concise."
)

# List all sessions
all_contexts = manager.list_all_contexts()
# Returns: {"cli": [{"session_id": "main", "date": "2025-11-06", ...}], ...}

# Get session stats
stats = manager.get_stats("cli:main")
# Returns: {"context_id": "cli:main", "message_count": 10, ...}

# Clear history (keeps session)
manager.clear_context("cli:main")

# Delete session entirely
manager.delete_context("cli:main")
```

### CLI Usage

```bash
# Interactive session picker
python chat.py

# Use specific session
python chat.py --session-id main

# List all sessions
python chat.py --list-sessions

# Delete a session
python chat.py --delete-session main
```

## Session Lifecycle

1. **Creation**: New session created on first message
   - Assigned today's date
   - Saved to `sessions/{type}/{date}/{id}.json`

2. **Usage**: Messages added during conversation
   - Auto-saved after each interaction
   - `last_active` updated

3. **Resume**: Load existing session
   - Searches across dates if needed
   - Full history restored

4. **Archive**: Old sessions remain on disk
   - Organized by date for easy cleanup
   - Can manually archive/delete old dates

## Migration

To migrate from flat structure (pre-date organization):

```bash
# Dry run (preview changes)
python scripts/migrate_sessions.py --dry-run

# Actual migration
python scripts/migrate_sessions.py

# Migrate custom directory
python scripts/migrate_sessions.py --storage-dir ./my_sessions
```

The migration script:
- Reads `created_at` from metadata
- Creates date subdirectories
- Moves sessions to appropriate date folders
- Updates metadata with `session_date` field

## Best Practices

### Session IDs

- **CLI**: Use descriptive names (`main`, `project_x`, `brainstorm`)
- **IRC**: Use channel names (`#python`, `#linux`)
- **Games**: Use game+run (`pokemon_run1`, `chess_game2`)
- **Web**: Use session tokens (`session_abc123`)

### Cleanup

- Archive old date folders periodically
- Delete unused sessions with `--delete-session`
- Keep active sessions, archive others

### Storage Location

- Default: `./sessions/` (relative to project root)
- Custom: Use `--storage-dir` or `storage_dir` parameter
- Production: Consider absolute paths

### Backup

- Sessions are just JSON files
- Easy to backup: `tar -czf sessions-backup.tar.gz sessions/`
- Version control: `.gitignore` should exclude `sessions/`

## Multi-Context Architecture

This storage system supports the Terrarium Agent's multi-context architecture:

- **Multiple endpoints** share one vLLM instance
- **Sequential processing**: One context active at a time
- **Fast switching**: In-memory cache + vLLM APC
- **Isolation**: Each context has independent history

Example flow:
```
IRC bot gets message → manager.process_with_context("irc:#python", msg)
Pokemon needs move   → manager.process_with_context("game:pokemon", state)
User asks question   → manager.process_with_context("cli:main", question)
```

Each maintains separate conversation history, saved to disk.

## Future Enhancements

Planned improvements:
- SQLite backend option (queryable, analytics)
- Automatic archival (move old sessions to archive/)
- Session search by content
- Export to different formats
- Session statistics dashboard

## Notes

- **Thread safety**: Not currently thread-safe (use for sequential processing)
- **File locking**: No locking (single-process assumption)
- **Cache size**: Default 10 sessions in memory (configurable)
- **Auto-save**: Enabled by default (can disable per-session)

## See Also

- `CLAUDE.md` - Project overview
- `agent/session_manager.py` - PersistentSession implementation
- `agent/multi_context_manager.py` - MultiContextManager implementation
- `scripts/migrate_sessions.py` - Migration utility
