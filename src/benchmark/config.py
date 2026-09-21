from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    warmup: int = 20
    rounds: int = 100
    batch_size: int = 1
    interleave: bool = False
    alternate_order: bool = True
    percentiles: tuple[float, ...] = (0.2, 0.8)
    repeat: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.repeat is not None:
            self.rounds = self.repeat
            self.repeat = None
