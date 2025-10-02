"""Plugin registry for custom transformers and policies."""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    """Simple plugin registry for custom transformers or policies."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, obj: Callable[..., Any]) -> None:
        """Register a plugin."""
        if name in self._registry:
            raise ValueError(f"Plugin '{name}' already registered.")
        self._registry[name] = obj

    def get(self, name: str) -> Callable[..., Any]:
        """Get a registered plugin."""
        if name not in self._registry:
            raise KeyError(f"Plugin '{name}' not found.")
        return self._registry[name]

    def available(self) -> list[str]:
        """List available plugin names."""
        return sorted(self._registry.keys())


# Global registry instance
GLOBAL_REGISTRY = Registry()
