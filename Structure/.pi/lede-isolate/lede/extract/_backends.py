"""Backend registry for lede enrichment primitives.

Primitives (metadata, phrases, correlate_facts) can dispatch to pluggable
backends. The 'regex' backend is always registered by lede itself. Other
backends (e.g. 'spacy' via the lede-spacy companion package) register
themselves at import time by calling `register(backend, primitive, fn)`.

The default backend is 'regex' to preserve lede's zero-dep identity.
Users opt in to 'auto' or 'spacy' via `lede.set_default_backend(name)`
or per-call `backend=` kwargs.

See docs/lede-spacy-integration.md.
"""
from __future__ import annotations
from typing import Callable, Any

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "regex": {},
}
_DEFAULT: str = "regex"


def register(backend: str, primitive: str, fn: Callable[..., Any]) -> None:
    """Register `fn` as the implementation of `primitive` under `backend`."""
    _REGISTRY.setdefault(backend, {})[primitive] = fn


def resolve(backend: str, primitive: str) -> Callable[..., Any]:
    """Return the registered callable for (backend, primitive).

    backend='auto' falls through to 'spacy' then 'regex' — useful for
    libraries that want the best available without hard-coding.
    """
    if backend == "auto":
        for candidate in ("spacy", "regex"):
            fn = _REGISTRY.get(candidate, {}).get(primitive)
            if fn is not None:
                return fn
        raise ImportError(
            f"no backend registered for primitive {primitive!r}"
        )
    try:
        return _REGISTRY[backend][primitive]
    except KeyError as e:
        known = sorted(_REGISTRY.keys())
        raise ImportError(
            f"backend={backend!r} not registered for primitive {primitive!r}. "
            f"Known backends: {known}. "
            f"Install lede-spacy to add the 'spacy' backend."
        ) from e


def get_default_backend() -> str:
    return _DEFAULT


def set_default_backend(name: str) -> None:
    """Set the process-wide default backend. Per-call `backend=` still wins.

    Valid names: 'regex' (always available), 'auto' (picks spacy if
    registered else regex), or any backend name that has been registered.
    """
    global _DEFAULT
    if name not in ("regex", "auto") and name not in _REGISTRY:
        known = sorted(_REGISTRY.keys())
        raise ValueError(
            f"unknown backend {name!r}. Known: {known + ['auto']}"
        )
    _DEFAULT = name
