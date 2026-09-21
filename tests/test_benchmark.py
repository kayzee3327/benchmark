from __future__ import annotations

from benchmark import (
    BenchmarkConfig,
    Reference,
    ReferenceResult,
    WallClockMeasurer,
    benchmark,
)


def test_target_without_references():
    result = benchmark(
        target=lambda: 42,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=2, rounds=5),
    )
    assert result.references == []
    assert result.target.measurement is not None
    assert len(result.target.measurement.times_ms) == 5


def test_one_reference_pass():
    result = benchmark(
        target=lambda: 10,
        references=[Reference("same", lambda: 10)],
        comparator=lambda actual, expected: None,
        config=BenchmarkConfig(warmup=1, rounds=3),
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
        config=BenchmarkConfig(warmup=1, rounds=3),
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
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    assert result.references[0].validation.passed is True
    assert result.references[1].validation.passed is False
    assert "999" in result.references[1].validation.error
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
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    assert result.references[0].validation.passed is True

    result2 = benchmark(
        target=lambda: 3.14,
        references=[Reference("far", lambda: 100.0)],
        comparator=approx,
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    assert result2.references[0].validation.passed is False


def test_wall_clock_measurement():
    result = benchmark(
        target=lambda: sum(range(1000)),
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=2, rounds=10),
    )
    m = result.target.measurement
    assert m is not None
    assert len(m.times_ms) == 10
    assert m.min_ms > 0
    assert m.min_ms <= m.median_ms


def test_no_measurer_returns_none():
    result = benchmark(
        target=lambda: 42,
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    assert result.target.measurement is None


def test_validation_pass_with_measurer():
    result = benchmark(
        target=lambda: 42,
        references=[Reference("ref", lambda: 42)],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=5),
    )
    assert result.target.measurement is not None
    assert len(result.target.measurement.times_ms) == 5


def test_references_also_measured():
    result = benchmark(
        target=lambda: 1,
        references=[
            Reference("a", lambda: 1),
            Reference("b", lambda: 1),
        ],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=5),
    )
    for ref in result.references:
        assert ref.measurement is not None
        assert ref.name in ("a", "b")
        assert len(ref.measurement.times_ms) == 5


def test_references_not_measured_on_failure():
    result = benchmark(
        target=lambda: 1,
        references=[
            Reference("ok", lambda: 1),
            Reference("bad", lambda: 2),
        ],
        comparator=lambda a, e: (_ for _ in ()).throw(ValueError("bad")) if a != e else None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    assert result.target.measurement is None
    for ref in result.references:
        assert ref.measurement is None


def test_output_fn_for_inplace_target():
    target_buf = [0]
    ref_buf = [0]

    def inplace_target():
        target_buf[0] = 99

    def inplace_ref():
        ref_buf[0] = 99

    result = benchmark(
        target=inplace_target,
        references=[
            Reference("ref", inplace_ref, output_fn=lambda: ref_buf[0]),
        ],
        comparator=lambda actual, expected: (
            None if actual == expected else (_ for _ in ()).throw(ValueError("mismatch"))
        ),
        output_fn=lambda: target_buf[0],
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    assert result.references[0].validation.passed is True


def test_output_fn_target_called_once_during_validation():
    call_count = [0]

    def counting_target():
        call_count[0] += 1

    result = benchmark(
        target=counting_target,
        references=[Reference("ref", lambda: None)],
        comparator=lambda a, e: None,
        output_fn=lambda: "result",
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    warmup_calls = 1
    validation_calls = 1
    measurement_warmup = 1
    measurement_repeat = 3
    assert call_count[0] == warmup_calls + validation_calls + measurement_warmup + measurement_repeat


# ---------- New tests for batched / interleaved / stats ----------


def test_measurement_result_fields():
    result = benchmark(
        target=lambda: sum(range(100)),
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=2, rounds=20),
    )
    m = result.target.measurement
    assert m is not None
    assert m.max_ms >= m.min_ms
    assert m.mad_ms >= 0
    assert "p20" in m.percentiles_ms
    assert "p80" in m.percentiles_ms
    assert m.percentiles_ms["p20"] <= m.percentiles_ms["p80"]


def test_batch_size():
    call_count = [0]

    def fn():
        call_count[0] += 1

    result = benchmark(
        target=fn,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=2, rounds=5, batch_size=3),
    )
    m = result.target.measurement
    assert m is not None
    assert len(m.times_ms) == 5
    warmup_calls = 1 + 2
    measurement_calls = 5 * 3
    assert call_count[0] == warmup_calls + measurement_calls


def test_comparison_result():
    result = benchmark(
        target=lambda: 1,
        references=[Reference("ref", lambda: 1)],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=10),
    )
    ref = result.references[0]
    assert isinstance(ref, ReferenceResult)
    assert ref.comparison is not None
    assert ref.comparison.reference_name == "ref"
    assert ref.comparison.median_speedup > 0


def test_interleaved_measurement():
    result = benchmark(
        target=lambda: sum(range(100)),
        references=[Reference("ref", lambda: sum(range(100)))],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=2, rounds=10, interleave=True),
    )
    ref = result.references[0]
    assert ref.measurement is not None
    assert ref.comparison is not None
    assert ref.comparison.paired_median_ratio is not None
    assert ref.comparison.ci_95_low is not None
    assert ref.comparison.ci_95_high is not None


def test_interleaved_with_batch():
    call_count_target = [0]
    call_count_ref = [0]

    def target_fn():
        call_count_target[0] += 1

    def ref_fn():
        call_count_ref[0] += 1

    result = benchmark(
        target=target_fn,
        references=[Reference("ref", ref_fn)],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(
            warmup=2, rounds=6, batch_size=4,
            interleave=True, alternate_order=True,
        ),
    )
    assert result.target.measurement is not None
    assert result.references[0].measurement is not None

    prep_target = 1
    prep_ref = 1
    validation_target = 1
    validation_ref = 1
    warmup_target = 2
    warmup_ref = 2
    measure_target = 6 * 4
    measure_ref = 6 * 4
    assert call_count_target[0] == prep_target + validation_target + warmup_target + measure_target
    assert call_count_ref[0] == prep_ref + validation_ref + warmup_ref + measure_ref


def test_custom_percentiles():
    result = benchmark(
        target=lambda: 42,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=20, percentiles=(0.1, 0.5, 0.9)),
    )
    m = result.target.measurement
    assert "p10" in m.percentiles_ms
    assert "p50" in m.percentiles_ms
    assert "p90" in m.percentiles_ms
    assert m.percentiles_ms["p10"] <= m.percentiles_ms["p50"] <= m.percentiles_ms["p90"]


def test_repeat_alias():
    cfg = BenchmarkConfig(warmup=1, repeat=50)
    assert cfg.rounds == 50

    cfg2 = BenchmarkConfig(warmup=1, rounds=7)
    assert cfg2.rounds == 7


def test_sequential_no_paired_ratio():
    result = benchmark(
        target=lambda: 1,
        references=[Reference("ref", lambda: 1)],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=5, interleave=False),
    )
    ref = result.references[0]
    assert ref.comparison is not None
    assert ref.comparison.paired_median_ratio is None
    assert ref.comparison.ci_95_low is None


def test_stats_compute_measurement():
    from benchmark.stats import compute_measurement
    m = compute_measurement([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m.min_ms == 1.0
    assert m.max_ms == 5.0
    assert m.median_ms == 3.0
    assert abs(m.mean_ms - 3.0) < 1e-9
    assert m.mad_ms == 1.0
    assert "p20" in m.percentiles_ms
    assert "p80" in m.percentiles_ms


def test_stats_compute_comparison_with_paired():
    from benchmark.stats import compute_comparison, compute_measurement
    t = compute_measurement([1.0] * 10)
    r = compute_measurement([2.0] * 10)
    c = compute_comparison(t, r, "ref", paired_ratios=[2.0] * 10)
    assert abs(c.median_speedup - 2.0) < 1e-9
    assert abs(c.paired_median_ratio - 2.0) < 1e-9
    assert c.ci_95_low is not None
    assert c.ci_95_high is not None


def test_print_result_no_crash(capsys):
    from benchmark import print_result
    result = benchmark(
        target=lambda: sum(range(100)),
        references=[Reference("ref", lambda: sum(range(100)))],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=5),
    )
    print_result(result)
    captured = capsys.readouterr()
    assert "Measurement" in captured.out
    assert "Comparison" in captured.out


def test_print_result_interleaved_no_crash(capsys):
    from benchmark import print_result
    result = benchmark(
        target=lambda: sum(range(100)),
        references=[Reference("ref", lambda: sum(range(100)))],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=10, interleave=True),
    )
    print_result(result)
    captured = capsys.readouterr()
    assert "paired" in captured.out
    assert "95% CI" in captured.out


def test_print_result_validation_failure_no_crash(capsys):
    from benchmark import print_result
    result = benchmark(
        target=lambda: 1,
        references=[Reference("bad", lambda: 2)],
        comparator=lambda a, e: (_ for _ in ()).throw(ValueError("mismatch")) if a != e else None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    print_result(result)
    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    assert "skipped" in captured.out


def test_print_results_table_no_crash(capsys):
    from benchmark import print_results_table
    r1 = benchmark(
        target=lambda: 1,
        references=[Reference("ref", lambda: 1)],
        comparator=lambda a, e: None,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    r2 = benchmark(
        target=lambda: 2,
        measurer=WallClockMeasurer(),
        config=BenchmarkConfig(warmup=1, rounds=3),
    )
    print_results_table([("bench1", r1), ("bench2", r2)])
    captured = capsys.readouterr()
    assert "bench1" in captured.out
    assert "bench2" in captured.out
