"""Dict-based strategy registry for equity and options."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategies.base import BaseStrategy

_registry: dict[str, type[BaseStrategy]] = {}


def register(strategy_cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Class decorator that registers a strategy by its ``name`` attribute."""
    _registry[strategy_cls.name] = strategy_cls
    return strategy_cls


def get(name: str) -> type[BaseStrategy]:
    """Lookup strategy class by name. Raises KeyError if not found."""
    return _registry[name]


def list_strategies() -> list[str]:
    """Return all registered strategy names."""
    return list(_registry.keys())


# Options strategy registry
_options_registry: dict[str, type] = {}


def register_options(strategy_cls: type) -> type:
    """Class decorator that registers an options strategy by ``name``."""
    _options_registry[strategy_cls.name] = strategy_cls
    return strategy_cls


def get_options(name: str) -> type:
    """Lookup options strategy class by name."""
    return _options_registry[name]


def list_options_strategies() -> list[str]:
    """Return all registered options strategy names."""
    return list(_options_registry.keys())
