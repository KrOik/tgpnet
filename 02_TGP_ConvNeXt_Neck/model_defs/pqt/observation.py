from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

import torch

ObservationSink = Callable[[str, torch.Tensor], None]
_SINK: ContextVar[ObservationSink | None] = ContextVar("pqt_observation_sink", default=None)


def publish(name: str, value: torch.Tensor) -> None:
    """Publish an ephemeral tensor to the active observer without retaining it."""
    sink = _SINK.get()
    if sink is not None:
        sink(name, value)


@contextmanager
def observation_sink(sink: ObservationSink | None) -> Iterator[None]:
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)
