"""Session management with persistent storage for multi-context conversations."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class PersistentSession:
    """
    Manages a chat session with persistent storage.

    Designed for multi-context architecture where many endpoints
    (IRC, game harnesses, web, CLI) maintain separate conversation contexts.
    """

    def __init__(
        self,
        session_id: str,
        context_type: str = "cli",
        storage_dir: Path = None,
        system_prompt: str = None,
        auto_save: bool = True
    ):
        """
        Initialize persistent session.

        Args:
            session_id: Unique identifier (e.g., '#python', 'pokemon_run1', 'user_session')
            context_type: Type of context ('irc', 'game', 'web', 'cli')
            storage_dir: Root directory for session storage (default: ./sessions)
            system_prompt: System prompt for the conversation
            auto_save: Whether to auto-save after each message
        """
        self.session_id = session_id
        self.context_type = context_type
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.auto_save = auto_save

        # Set up storage
        self.storage_dir = storage_dir or Path("sessions")
        self.session_dir = self.storage_dir / context_type
        self.session_file = self.session_dir / f"{session_id}.json"

        # Conversation state
        self.messages: List[Dict[str, str]] = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "message_count": 0,
            "session_id": session_id,
            "context_type": context_type
        }

        # Ensure storage directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def context_id(self) -> str:
        """Get full context identifier: {type}:{id}"""
        return f"{self.context_type}:{self.session_id}"

    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Update metadata
        self.metadata["last_active"] = datetime.now().isoformat()
        self.metadata["message_count"] = len(self.messages)

        # Auto-save if enabled
        if self.auto_save:
            self.save()

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get full conversation history including system prompt.

        Format suitable for vLLM chat API.

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        history = [{"role": "system", "content": self.system_prompt}]

        # Add messages without timestamp (API doesn't need it)
        for msg in self.messages:
            history.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return history

    def get_messages(self) -> List[Dict[str, str]]:
        """Get raw message history with timestamps (without system prompt)."""
        return self.messages.copy()

    def clear(self):
        """Clear conversation history (keeps metadata)."""
        self.messages = []
        self.metadata["message_count"] = 0
        self.metadata["last_active"] = datetime.now().isoformat()

        if self.auto_save:
            self.save()

    def save(self) -> bool:
        """
        Save session to disk.

        Returns:
            True if save successful, False otherwise
        """
        try:
            data = {
                "session_id": self.session_id,
                "context_type": self.context_type,
                "system_prompt": self.system_prompt,
                "messages": self.messages,
                "metadata": self.metadata
            }

            # Write to file with pretty printing
            self.session_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False)
            )

            return True

        except Exception as e:
            print(f"Error saving session {self.context_id}: {e}")
            return False

    def load(self) -> bool:
        """
        Load session from disk.

        Returns:
            True if load successful, False if session doesn't exist
        """
        if not self.session_file.exists():
            return False

        try:
            data = json.loads(self.session_file.read_text())

            # Restore session state
            self.session_id = data.get("session_id", self.session_id)
            self.context_type = data.get("context_type", self.context_type)
            self.system_prompt = data.get("system_prompt", self.system_prompt)
            self.messages = data.get("messages", [])
            self.metadata = data.get("metadata", self.metadata)

            # Update last_active
            self.metadata["last_active"] = datetime.now().isoformat()

            return True

        except Exception as e:
            print(f"Error loading session {self.context_id}: {e}")
            return False

    def delete(self) -> bool:
        """
        Delete session from disk.

        Returns:
            True if deletion successful
        """
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting session {self.context_id}: {e}")
            return False

    def get_stats(self) -> Dict:
        """
        Get session statistics.

        Returns:
            Dictionary with session stats
        """
        return {
            "context_id": self.context_id,
            "session_id": self.session_id,
            "context_type": self.context_type,
            "message_count": len(self.messages),
            "created_at": self.metadata.get("created_at"),
            "last_active": self.metadata.get("last_active"),
            "exists_on_disk": self.session_file.exists()
        }

    def __repr__(self):
        return f"PersistentSession({self.context_id}, messages={len(self.messages)})"
