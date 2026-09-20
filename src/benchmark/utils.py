from __future__ import annotations

from benchmark.result import BenchmarkResult, ImplementationResult


def print_result(result: BenchmarkResult) -> None:
    all_impls = [result.target] + result.references

    # --- Validation ---
    has_validations = any(r.validation is not None for r in result.references)
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


def _print_measurement_table(impls: list[ImplementationResult]) -> None:
    headers = ["Name", "Min (ms)", "Median (ms)", "Mean (ms)"]
    rows: list[list[str]] = []
    for impl in impls:
        m = impl.measurement
        rows.append([
            impl.name,
            f"{m.min_ms:.4f}",
            f"{m.median_ms:.4f}",
            f"{m.mean_ms:.4f}",
        ])

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

    # --- Speedup vs target ---
    if len(impls) > 1:
        target_median = impls[0].measurement.median_ms
        print("Speedup (target / reference median)")
        print("-" * 40)
        for impl in impls[1:]:
            ref_median = impl.measurement.median_ms
            if ref_median > 0:
                ratio = ref_median / target_median
                print(f"  vs {impl.name}: {ratio:.2f}x")
        print()


def print_results_table(
    named_results: list[tuple[str, BenchmarkResult]],
) -> None:
    PASS = "\033[32mPASS\033[0m"
    FAIL = "\033[31mFAIL\033[0m"
    NA = "-"

    has_refs = any(r.references for _, r in named_results)

    headers = ["Benchmark"]
    if has_refs:
        headers.append("Valid")
    headers += ["Target (ms)", "Ref (ms)", "Speedup"] if has_refs else ["Time (ms)"]

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
        else:
            row.append(NA)

        if has_refs:
            if result.references and result.references[0].measurement is not None:
                ref_m = result.references[0].measurement
                row.append(f"{ref_m.median_ms:.4f}")
                if target_m is not None and target_m.median_ms > 0:
                    speedup = ref_m.median_ms / target_m.median_ms
                    row.append(f"{speedup:.2f}x")
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
