from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str, item: T | None = None) -> Callable[[T], T] | T:
        if item is not None:
            self._items[key] = item
            return item

        def decorator(value: T) -> T:
            self._items[key] = value
            return value

        return decorator

    def get(self, key: str) -> T:
        if key not in self._items:
            raise KeyError(f"{key!r} is not registered in {self.name}.")
        return self._items[key]

    def keys(self) -> tuple[str, ...]:
        return tuple(self._items.keys())
