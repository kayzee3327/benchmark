from __future__ import annotations

from benchmark.result import BenchmarkResult, ImplementationResult, MeasurementResult, ReferenceResult


def print_result(result: BenchmarkResult) -> None:
    all_impls: list[ImplementationResult] = [result.target] + result.references

    # --- Validation ---
    has_validations = any(
        isinstance(r, ReferenceResult) and r.validation is not None
        for r in result.references
    )
    if has_validations:
        print("Validation")
        print("-" * 40)
        for ref in result.references:
            v = ref.validation
            if v is None:
                continue
            status = "\033[32mPASS\033[0m" if v.passed else "\033[31mFAIL\033[0m"
            print(f"  {v.reference_name}: {status}")
            if v.error:
                first_line = v.error.splitlines()[0]
                print(f"    {first_line}")
            if v.diff_summary:
                print(v.diff_summary)
        print()

    # --- Measurement ---
    measured = [impl for impl in all_impls if impl.measurement is not None]
    if not measured:
        if has_validations and not all(
            r.validation.passed for r in result.references if r.validation
        ):
            print("Measurement skipped (validation failed).")
        else:
            print("No measurement data.")
        return

    _print_measurement_table(measured)
    _print_comparison(result)


def _print_measurement_table(impls: list[ImplementationResult]) -> None:
    pct_keys: list[str] = []
    for impl in impls:
        if impl.measurement and impl.measurement.percentiles_ms:
            pct_keys = sorted(impl.measurement.percentiles_ms.keys())
            break

    headers = ["Name", "Min (ms)", "Median (ms)", "Mean (ms)", "Max (ms)", "MAD (ms)"]
    for key in pct_keys:
        headers.append(f"{key} (ms)")

    rows: list[list[str]] = []
    for impl in impls:
        m = impl.measurement
        if m is None:
            continue
        row = [
            impl.name,
            f"{m.min_ms:.4f}",
            f"{m.median_ms:.4f}",
            f"{m.mean_ms:.4f}",
            f"{m.max_ms:.4f}",
            f"{m.mad_ms:.4f}",
        ]
        for key in pct_keys:
            row.append(f"{m.percentiles_ms.get(key, 0):.4f}")
        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i == 0:
                parts.append(cell.ljust(col_widths[i]))
            else:
                parts.append(cell.rjust(col_widths[i]))
        return "  ".join(parts)

    print("Measurement")
    print("-" * 40)
    print(f"  {fmt_row(headers)}")
    print(f"  {'  '.join('-' * w for w in col_widths)}")
    for row in rows:
        print(f"  {fmt_row(row)}")
    print()


def _print_comparison(result: BenchmarkResult) -> None:
    refs_with_cmp = [
        r for r in result.references
        if isinstance(r, ReferenceResult) and r.comparison is not None
    ]
    if not refs_with_cmp:
        return

    print("Comparison (target / reference)")
    print("-" * 40)
    for ref in refs_with_cmp:
        c = ref.comparison
        line = f"  vs {c.reference_name}: {c.median_speedup:.2f}x (median ratio)"
        if c.paired_median_ratio is not None:
            line += f"  paired: {c.paired_median_ratio:.2f}x"
        if c.ci_95_low is not None and c.ci_95_high is not None:
            line += f"  95% CI [{c.ci_95_low:.2f}x, {c.ci_95_high:.2f}x]"
        print(line)
    print()


def print_results_table(
    named_results: list[tuple[str, BenchmarkResult]],
) -> None:
    PASS = "\033[32mPASS\033[0m"
    FAIL = "\033[31mFAIL\033[0m"
    NA = "-"

    has_refs = any(r.references for _, r in named_results)
    has_paired = any(
        isinstance(ref, ReferenceResult)
        and ref.comparison is not None
        and ref.comparison.paired_median_ratio is not None
        for _, r in named_results
        for ref in r.references
    )

    pct_keys: list[str] = []
    for _, result in named_results:
        m = result.target.measurement
        if m is not None and m.percentiles_ms:
            pct_keys = sorted(m.percentiles_ms.keys())
            break

    headers = ["Benchmark"]
    if has_refs:
        headers.append("Valid")
    if has_refs:
        headers += ["Target (ms)", "MAD"]
        for key in pct_keys:
            headers.append(key)
        headers += ["Ref (ms)", "MAD"]
        for key in pct_keys:
            headers.append(f"r.{key}")
        headers.append("Speedup")
        if has_paired:
            headers += ["Paired", "95% CI"]
    else:
        headers += ["Time (ms)", "MAD"]
        for key in pct_keys:
            headers.append(key)

    rows: list[list[str]] = []
    for name, result in named_results:
        target_m = result.target.measurement
        row = [name]

        if has_refs:
            if result.references:
                all_passed = all(
                    r.validation.passed
                    for r in result.references
                    if r.validation is not None
                )
                row.append(PASS if all_passed else FAIL)
            else:
                row.append(NA)

        if target_m is not None:
            row.append(f"{target_m.median_ms:.4f}")
            row.append(f"{target_m.mad_ms:.4f}")
            for key in pct_keys:
                row.append(f"{target_m.percentiles_ms.get(key, 0):.4f}")
        else:
            row.append(NA)
            row.append(NA)
            for _ in pct_keys:
                row.append(NA)

        if has_refs:
            ref = result.references[0] if result.references else None
            ref_m = ref.measurement if ref else None
            if ref_m is not None:
                row.append(f"{ref_m.median_ms:.4f}")
                row.append(f"{ref_m.mad_ms:.4f}")
                for key in pct_keys:
                    row.append(f"{ref_m.percentiles_ms.get(key, 0):.4f}")
                if target_m is not None and target_m.median_ms > 0:
                    speedup = ref_m.median_ms / target_m.median_ms
                    row.append(f"{speedup:.2f}x")
                else:
                    row.append(NA)
            else:
                row += [NA, NA] + [NA] * len(pct_keys) + [NA]

            if has_paired:
                cmp = (
                    ref.comparison
                    if ref is not None and isinstance(ref, ReferenceResult)
                    else None
                )
                if cmp is not None and cmp.paired_median_ratio is not None:
                    row.append(f"{cmp.paired_median_ratio:.2f}x")
                    if cmp.ci_95_low is not None and cmp.ci_95_high is not None:
                        row.append(f"[{cmp.ci_95_low:.2f}, {cmp.ci_95_high:.2f}]")
                    else:
                        row.append(NA)
                else:
                    row += [NA, NA]

        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            visible = cell.replace("\033[32m", "").replace("\033[31m", "").replace("\033[0m", "")
            col_widths[i] = max(col_widths[i], len(visible))

    def _visible_len(s: str) -> int:
        return len(s.replace("\033[32m", "").replace("\033[31m", "").replace("\033[0m", ""))

    def fmt_cell(cell: str, width: int, align_left: bool) -> str:
        pad = width - _visible_len(cell)
        if align_left:
            return cell + " " * pad
        return " " * pad + cell

    sep = "  "
    header_line = sep.join(
        fmt_cell(h, col_widths[i], i == 0) for i, h in enumerate(headers)
    )
    divider = sep.join("-" * w for w in col_widths)

    print(header_line)
    print(divider)
    for row in rows:
        print(sep.join(
            fmt_cell(row[i], col_widths[i], i == 0) for i in range(len(headers))
        ))

    failed = []
    for name, result in named_results:
        for ref in result.references:
            if ref.validation is not None and not ref.validation.passed:
                failed.append((name, ref))
    if failed:
        print()
        print("Validation errors:")
        for bench_name, ref in failed:
            first_line = ref.validation.error.splitlines()[0] if ref.validation.error else "unknown"
            print(f"  {bench_name} vs {ref.name}: {first_line}")
            if ref.validation.diff_summary:
                print(ref.validation.diff_summary)

    print()
    legend: list[str] = [
        "Target/Ref (ms) : median latency per call",
        "MAD             : median absolute deviation, median(|xi - median|), measures spread",
    ]
    for key in pct_keys:
        pct_val = int(key[1:])
        legend.append(f"{key:<16}: {pct_val}th percentile latency of the target")
    if has_refs:
        for key in pct_keys:
            pct_val = int(key[1:])
            legend.append(f"r.{key:<14}: {pct_val}th percentile latency of the reference")
        legend.append(
            "Speedup         : median(ref) / median(target), >1 means target is faster"
        )
    if has_paired:
        legend.append(
            "Paired          : median of per-round ratios, median(ref_i / target_i)"
        )
        legend.append(
            "95% CI          : bootstrap 95% confidence interval on the paired"
            " median ratio (10k resamples)"
        )
    print("\n".join(legend))
