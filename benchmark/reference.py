from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Reference:
    name: str
    fn: Callable[[], Any]
