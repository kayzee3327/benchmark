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
