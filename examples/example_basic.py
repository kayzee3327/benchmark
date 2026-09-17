"""Example: benchmarking a simple Python function with the benchmark framework."""

from benchmark import (
    BenchmarkConfig,
    Reference,
    WallClockMeasurer,
    benchmark,
    print_result,
)


def target_impl(n: int) -> list[int]:
    return sorted(range(n, 0, -1))


def reference_impl(n: int) -> list[int]:
    return list(range(1, n + 1))


def wrong_impl(n: int) -> list[int]:
    return list(range(n))


def main():
    n = 5000

    result = benchmark(
        target=lambda: target_impl(n),
        references=[
            Reference("reference", lambda: reference_impl(n)),
            Reference("wrong", lambda: wrong_impl(n)),
        ],
        comparator=lambda actual, expected: (
            None if actual == expected else (_ for _ in ()).throw(AssertionError("mismatch"))
        ),
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=5, repeat=50),
    )

    print_result(result)


if __name__ == "__main__":
    main()
