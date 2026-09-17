from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    warmup: int = 20
    repeat: int = 100
