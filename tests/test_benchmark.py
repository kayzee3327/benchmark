from __future__ import annotations

from benchmark import (
    BenchmarkConfig,
    Reference,
    WallClockMeasurer,
    benchmark,
)


def test_target_without_references():
    result = benchmark(
        target=lambda: 42,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=2, repeat=5),
    )
    assert result.references == []
    assert result.target.measurement is not None
    assert len(result.target.measurement.times_ms) == 5


def test_one_reference_pass():
    result = benchmark(
        target=lambda: 10,
        references=[Reference("same", lambda: 10)],
        comparator=lambda actual, expected: None,
        config=BenchmarkConfig(warmup=1, repeat=3),
    )
    assert len(result.references) == 1
    assert result.references[0].validation.passed is True
    assert result.references[0].validation.reference_name == "same"


def test_multiple_references():
    result = benchmark(
        target=lambda: 5,
        references=[
            Reference("a", lambda: 5),
            Reference("b", lambda: 5),
            Reference("c", lambda: 5),
        ],
        comparator=lambda actual, expected: None,
        config=BenchmarkConfig(warmup=1, repeat=3),
    )
    assert len(result.references) == 3
    assert all(r.validation.passed for r in result.references)


def test_validation_failure():
    def strict_comparator(actual, expected):
        if actual != expected:
            raise ValueError(f"{actual} != {expected}")

    result = benchmark(
        target=lambda: 1,
        references=[
            Reference("good", lambda: 1),
            Reference("bad", lambda: 999),
        ],
        comparator=strict_comparator,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, repeat=3),
    )
    assert result.references[0].validation.passed is True
    assert result.references[1].validation.passed is False
    assert "999" in result.references[1].validation.error
    # measurement should be skipped for all when any validation fails
    assert result.target.measurement is None
    assert result.references[0].measurement is None
    assert result.references[1].measurement is None


def test_custom_comparator():
    def approx(actual, expected):
        if abs(actual - expected) > 0.5:
            raise AssertionError(f"Not close enough: {actual} vs {expected}")

    result = benchmark(
        target=lambda: 3.14,
        references=[Reference("close", lambda: 3.0)],
        comparator=approx,
    )
    assert result.references[0].validation.passed is True

    result2 = benchmark(
        target=lambda: 3.14,
        references=[Reference("far", lambda: 100.0)],
        comparator=approx,
    )
    assert result2.references[0].validation.passed is False


def test_wall_clock_measurement():
    import time

    result = benchmark(
        target=lambda: time.sleep(0.001),
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, repeat=5),
    )
    m = result.target.measurement
    assert m is not None
    assert m.min_ms > 0
    assert m.median_ms >= m.min_ms
    assert len(m.times_ms) == 5


def test_no_measurer_returns_none():
    result = benchmark(target=lambda: 42)
    assert result.target.measurement is None


def test_validation_pass_with_measurer():
    result = benchmark(
        target=lambda: 7,
        references=[Reference("ref", lambda: 7)],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, repeat=3),
    )
    assert all(r.validation.passed for r in result.references)
    assert result.target.measurement is not None


def test_references_also_measured():
    call_counts = {"target": 0, "ref_a": 0, "ref_b": 0}

    def target_fn():
        call_counts["target"] += 1
        return 1

    def ref_a_fn():
        call_counts["ref_a"] += 1
        return 1

    def ref_b_fn():
        call_counts["ref_b"] += 1
        return 1

    result = benchmark(
        target=target_fn,
        references=[
            Reference("ref_a", ref_a_fn),
            Reference("ref_b", ref_b_fn),
        ],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=2, repeat=5),
    )
    assert result.target.measurement is not None
    assert result.references[0].measurement is not None
    assert result.references[1].measurement is not None
    assert len(result.references[0].measurement.times_ms) == 5
    assert len(result.references[1].measurement.times_ms) == 5
    assert result.references[0].name == "ref_a"
    assert result.references[1].name == "ref_b"


def test_references_not_measured_on_failure():
    def strict_comparator(actual, expected):
        if actual != expected:
            raise ValueError("mismatch")

    result = benchmark(
        target=lambda: 1,
        references=[Reference("bad", lambda: 2)],
        comparator=strict_comparator,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, repeat=3),
    )
    assert result.references[0].validation.passed is False
    assert result.target.measurement is None
    assert result.references[0].measurement is None
