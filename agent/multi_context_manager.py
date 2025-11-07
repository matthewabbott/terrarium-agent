"""Multi-context manager for coordinating context swapping across endpoints."""

from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict

from agent.session_manager import PersistentSession
from llm.vllm_client import VLLMClient


class MultiContextManager:
    """
    Manages multiple conversation contexts with fast switching.

    Designed for sequential context swapping where multiple endpoints
    (IRC, game harnesses, web, CLI) share a single vLLM instance.

    Features:
    - In-memory cache for fast context switching
    - Lazy loading (load from disk only when needed)
    - Auto-save after each interaction
    - LRU eviction for memory management
    """

    def __init__(
        self,
        vllm_client: VLLMClient,
        storage_dir: Path = None,
        cache_size: int = 10
    ):
        """
        Initialize multi-context manager.

        Args:
            vllm_client: VLLMClient instance for LLM interactions
            storage_dir: Root directory for session storage
            cache_size: Maximum number of contexts to keep in memory (LRU)
        """
        self.vllm_client = vllm_client
        self.storage_dir = storage_dir or Path("sessions")
        self.cache_size = cache_size

        # In-memory cache (LRU): context_id -> PersistentSession
        self._cache: OrderedDict[str, PersistentSession] = OrderedDict()

        # Currently active context
        self._active_context_id: Optional[str] = None

    def _parse_context_id(self, context_id: str) -> tuple[str, str]:
        """
        Parse context ID into type and session ID.

        Args:
            context_id: Full context ID (e.g., 'irc:#python')

        Returns:
            Tuple of (context_type, session_id)

        Raises:
            ValueError: If context ID format is invalid
        """
        parts = context_id.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid context_id format: '{context_id}'. "
                "Expected format: 'type:id' (e.g., 'irc:#python')"
            )
        return parts[0], parts[1]

    def get_or_create_session(
        self,
        context_id: str,
        system_prompt: Optional[str] = None
    ) -> PersistentSession:
        """
        Get session from cache or create/load it.

        Args:
            context_id: Full context ID (e.g., 'irc:#python', 'game:pokemon')
            system_prompt: System prompt (only used if creating new session)

        Returns:
            PersistentSession for the context
        """
        # Check cache first
        if context_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(context_id)
            return self._cache[context_id]

        # Parse context ID
        context_type, session_id = self._parse_context_id(context_id)

        # Create new session
        session = PersistentSession(
            session_id=session_id,
            context_type=context_type,
            storage_dir=self.storage_dir,
            system_prompt=system_prompt,
            auto_save=True
        )

        # Try to load from disk
        session.load()  # Returns False if doesn't exist, that's okay

        # Add to cache
        self._cache[context_id] = session

        # Evict oldest if cache is full
        if len(self._cache) > self.cache_size:
            # Remove least recently used (first item)
            old_context_id, old_session = self._cache.popitem(last=False)
            # Ensure it's saved before eviction
            old_session.save()

        return session

    async def process_with_context(
        self,
        context_id: str,
        user_message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Process message with specific context (main entry point for endpoints).

        This method:
        1. Activates the context (loads if needed)
        2. Adds user message to history
        3. Generates response from vLLM
        4. Adds response to history
        5. Auto-saves context

        Args:
            context_id: Full context ID (e.g., 'irc:#python')
            user_message: User's message
            system_prompt: System prompt (only used if creating new session)
            **kwargs: Additional arguments to pass to vLLM (temperature, etc.)

        Returns:
            Assistant's response

        Raises:
            Exception: If generation fails
        """
        # Get or create session
        session = self.get_or_create_session(context_id, system_prompt)

        # Add user message
        session.add_message("user", user_message)

        # Get conversation history
        messages = session.get_history()

        # Generate response using vLLM
        response = await self.vllm_client.chat(messages=messages, **kwargs)

        # Add assistant response
        session.add_message("assistant", response)

        # Mark as active
        self._active_context_id = context_id

        return response

    def get_active_context_id(self) -> Optional[str]:
        """Get currently active context ID."""
        return self._active_context_id

    def get_active_session(self) -> Optional[PersistentSession]:
        """Get currently active session."""
        if self._active_context_id:
            return self._cache.get(self._active_context_id)
        return None

    def list_cached_contexts(self) -> List[str]:
        """Get list of context IDs currently in memory cache."""
        return list(self._cache.keys())

    def list_all_contexts(self) -> Dict[str, List[Dict]]:
        """
        List all persisted contexts from disk (date-based organization).

        Returns:
            Dict mapping context_type to list of session info dicts.
            Each session dict contains: session_id, date, file_path

        Example:
            {
                "cli": [
                    {"session_id": "xyz123", "date": "2025-11-06", "path": "sessions/cli/2025-11-06/xyz123.json"},
                    {"session_id": "main", "date": "2025-11-05", "path": "sessions/cli/2025-11-05/main.json"}
                ],
                "irc": [...]
            }
        """
        contexts = {}

        if not self.storage_dir.exists():
            return contexts

        # Iterate through context type directories
        for type_dir in self.storage_dir.iterdir():
            if not type_dir.is_dir():
                continue

            context_type = type_dir.name
            sessions = []

            # Iterate through date subdirectories (date-based organization)
            for date_dir in type_dir.iterdir():
                if not date_dir.is_dir():
                    # Handle legacy flat structure (sessions without dates)
                    if date_dir.suffix == ".json":
                        sessions.append({
                            "session_id": date_dir.stem,
                            "date": "unknown",
                            "path": str(date_dir)
                        })
                    continue

                # Find all JSON files in this date directory
                for session_file in date_dir.glob("*.json"):
                    session_id = session_file.stem
                    sessions.append({
                        "session_id": session_id,
                        "date": date_dir.name,
                        "path": str(session_file)
                    })

            if sessions:
                # Sort by date (newest first), then by session_id
                sessions.sort(key=lambda x: (x["date"], x["session_id"]), reverse=True)
                contexts[context_type] = sessions

        return contexts

    def clear_context(self, context_id: str):
        """
        Clear conversation history for a context (keeps session).

        Args:
            context_id: Context to clear
        """
        session = self.get_or_create_session(context_id)
        session.clear()

    def delete_context(self, context_id: str):
        """
        Delete a context entirely (from cache and disk).

        Args:
            context_id: Context to delete
        """
        # Remove from cache
        if context_id in self._cache:
            session = self._cache.pop(context_id)
            session.delete()
        else:
            # Not in cache, load and delete
            context_type, session_id = self._parse_context_id(context_id)
            session = PersistentSession(
                session_id=session_id,
                context_type=context_type,
                storage_dir=self.storage_dir
            )
            session.delete()

        # Clear active if it was active
        if self._active_context_id == context_id:
            self._active_context_id = None

    def save_all(self):
        """Save all cached sessions to disk."""
        for session in self._cache.values():
            session.save()

    def get_stats(self, context_id: str) -> Optional[Dict]:
        """
        Get statistics for a specific context.

        Args:
            context_id: Context to get stats for

        Returns:
            Stats dictionary or None if context doesn't exist
        """
        session = self.get_or_create_session(context_id)
        return session.get_stats()

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "active_context": self._active_context_id,
            "cached_contexts": list(self._cache.keys())
        }

    def __repr__(self):
        return f"MultiContextManager(cached={len(self._cache)}/{self.cache_size}, active={self._active_context_id})"
