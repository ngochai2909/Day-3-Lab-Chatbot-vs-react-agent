import os
import threading
from typing import List, Optional


class KeyManager:
    """
    Manages a pool of API keys and rotates through them.

    Reads keys from environment variables in two supported formats:
    1. Comma-separated:  OPENAI_API_KEYS=key1,key2,key3
    2. Numbered suffix:  OPENAI_API_KEY, OPENAI_API_KEY_2, OPENAI_API_KEY_3, ...

    Use `current()` to get the active key and `rotate()` to advance to the next
    key (e.g. after hitting a rate limit / quota error).
    """

    def __init__(self, provider_prefix: str):
        """
        Args:
            provider_prefix: e.g. "OPENAI" or "GEMINI".
        """
        self.provider_prefix = provider_prefix.upper()
        self._keys: List[str] = self._load_keys()
        self._index = 0
        self._lock = threading.Lock()

    def _load_keys(self) -> List[str]:
        keys: List[str] = []

        # Format 1: comma-separated list, e.g. OPENAI_API_KEYS=a,b,c
        plural = os.getenv(f"{self.provider_prefix}_API_KEYS")
        if plural:
            keys.extend(k.strip() for k in plural.split(",") if k.strip())

        # Format 2: single + numbered suffixes (KEY, KEY_2, KEY_3, ...)
        single = os.getenv(f"{self.provider_prefix}_API_KEY")
        if single and single.strip():
            keys.append(single.strip())

        i = 2
        while True:
            val = os.getenv(f"{self.provider_prefix}_API_KEY_{i}")
            if not val or not val.strip():
                break
            keys.append(val.strip())
            i += 1

        # De-duplicate while preserving order
        seen = set()
        unique_keys = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique_keys.append(k)
        return unique_keys

    @property
    def count(self) -> int:
        return len(self._keys)

    def current(self) -> Optional[str]:
        """Return the currently active key (or None if no keys configured)."""
        if not self._keys:
            return None
        return self._keys[self._index]

    def rotate(self) -> Optional[str]:
        """Advance to the next key and return it. Wraps around."""
        with self._lock:
            if not self._keys:
                return None
            self._index = (self._index + 1) % len(self._keys)
            return self._keys[self._index]
