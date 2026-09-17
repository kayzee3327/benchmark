# benchmark

A generic Python benchmark framework for operator optimization. Define what to run — the framework validates correctness and measures performance.

## Project structure

```
benchmark/
├── src/
│   └── benchmark/       # package source
│       ├── measurers/   # built-in measurers (cpu, npu)
│       ├── benchmark.py
│       ├── config.py
│       ├── measurement.py
│       ├── reference.py
│       ├── result.py
│       ├── utils.py
│       └── validation.py
├── examples/
├── tests/
└── pyproject.toml
```

## Install

```bash
cd benchmark
pip install -e .
```

After this, `import benchmark` works from anywhere in the virtualenv.

## Quick start

```python
from benchmark import (
    BenchmarkConfig,
    Reference,
    WallClockMeasurer,
    benchmark,
    print_result,
)

target = lambda: sorted(range(1000, 0, -1))

references = [
    Reference("list_range", lambda: list(range(1, 1001))),
]

result = benchmark(
    target=target,
    references=references,
    comparator=lambda actual, expected: (
        None if actual == expected
        else (_ for _ in ()).throw(AssertionError("mismatch"))
    ),
    measurer=WallClockMeasurer(),
    config=BenchmarkConfig(warmup=10, repeat=50),
)

print_result(result)
```

Output:

```
Validation
----------------------------------------
  list_range: PASS

Measurement
----------------------------------------
  Name        Min (ms)  Median (ms)  Mean (ms)
  ----------  --------  -----------  ---------
  target        0.0135       0.0136     0.0137
  list_range    0.0104       0.0104     0.0106

Speedup (target / reference median)
----------------------------------------
  vs list_range: 0.76x
```

## Custom comparator

The default comparator uses `torch.testing.assert_close`. Pass your own for non-tensor outputs or custom tolerance:

```python
def my_comparator(actual, expected):
    if abs(actual - expected) > 1e-3:
        raise AssertionError(f"too far: {actual} vs {expected}")

result = benchmark(target=..., references=..., comparator=my_comparator)
```

## Custom measurer

`WallClockMeasurer` uses `time.perf_counter` — fine for CPU work, not for async device execution. For CUDA/NPU, subclass `Measurer`:

```python
from benchmark import Measurer, MeasurementResult, BenchmarkConfig
import statistics

class CudaEventMeasurer(Measurer):
    def measure(self, fn, config):
        import torch
        for _ in range(config.warmup):
            fn()
            torch.cuda.synchronize()
        times = []
        for _ in range(config.repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return MeasurementResult(
            times_ms=times,
            min_ms=min(times),
            median_ms=statistics.median(times),
            mean_ms=statistics.mean(times),
        )
```

## Run tests

```bash
cd /path/to/benchmark
python -m pytest tests/ -v
```
