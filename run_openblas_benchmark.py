#LD_LIBRARY_PATH=. python run_openblas_benchmark.py
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
RESULT_DIR = PROJECT_DIR / "openblas_result"
RESULT_CSV = RESULT_DIR / "openblas_results.csv"
AVERAGE_CSV = RESULT_DIR / "openblas_averages.csv"
PLOT_FILE = RESULT_DIR / "openblas_plot.png"

SIZES = [16, 128, 1024, 8192]
RUNS_PER_SIZE = 10

DURATION_PATTERNS = (
    re.compile(r"^duration\s*=\s*(\d+)ns\s*$", re.IGNORECASE),
    re.compile(r"^improved\s+duration\s*=\s*(\d+)ns\s*$", re.IGNORECASE),
    re.compile(r"^openblas\s+duration\s*=\s*(\d+)ns\s*$", re.IGNORECASE),
)


def resolve_exe(path_arg: str) -> Path:
    candidate = Path(path_arg)
    if candidate.is_absolute():
        return candidate
    return PROJECT_DIR / candidate


def default_executable_name(stem: str) -> str:
    if os.name == "nt":
        return f"{stem}.exe"
    return stem


def ensure_executable_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required executable not found: {path}")


def parse_duration_output(output: str) -> int:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        for pattern in DURATION_PATTERNS:
            match = pattern.match(line)
            if match is not None:
                return int(match.group(1))
    raise RuntimeError(f"Unexpected matrix program output:\n{output}")


def delete_input_dir() -> None:
    if not INPUT_DIR.exists():
        return

    try:
        shutil.rmtree(INPUT_DIR)
    except OSError as exc:
        print(f"Warning: failed to delete {INPUT_DIR}: {exc}", flush=True)


def run_cycle(size: int, rep: int, generator_exe: Path, ob_exe: Path) -> dict[str, int]:
    seed = int(time.time_ns() & 0xFFFFFFFF) ^ (size << 8) ^ rep

    subprocess.run(
        [str(generator_exe), str(size), str(seed)],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    ob_completed = subprocess.run(
        [str(ob_exe)],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    ob_duration_ns = parse_duration_output(ob_completed.stdout)

    return {
        "size": size,
        "rep": rep,
        "seed": seed,
        "ob_duration_ns": ob_duration_ns,
    }


def compute_averages(rows: list[dict[str, int]]) -> list[dict[str, float]]:
    grouped: dict[int, dict[str, int]] = {
        size: {"count": 0, "ob_total": 0, "ob3_total": 0} for size in SIZES
    }

    for row in rows:
        size = row["size"]
        grouped[size]["count"] += 1
        grouped[size]["ob_total"] += row["ob_duration_ns"]

    averages: list[dict[str, float]] = []
    for size in SIZES:
        count = grouped[size]["count"]
        if count == 0:
            raise RuntimeError(f"No benchmark rows collected for size {size}")

        averages.append(
            {
                "size": float(size),
                "ob_avg_ns": grouped[size]["ob_total"] / count,
            }
        )

    return averages


def save_rows_csv(rows: list[dict[str, int]]) -> None:
    fieldnames = ["size", "rep", "seed", "ob_duration_ns"]
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_averages_csv(averages: list[dict[str, float]]) -> None:
    fieldnames = ["size", "ob_avg_ns"]
    with AVERAGE_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(averages)


def draw_grouped_bar_chart(averages: list[dict[str, float]]) -> None:
    size_labels = [f"{int(item['size'])}x{int(item['size'])}" for item in averages]
    ob_values = [item["ob_avg_ns"] for item in averages]

    group_step = 1.8
    bar_width = 0.34
    x_centers = [index * group_step for index in range(len(averages))]
    ob_positions = [x - bar_width / 2 for x in x_centers]

    plt.figure(figsize=(10, 6))
    plt.bar(ob_positions, ob_values, width=bar_width, label="ob.exe average")
    plt.xticks(x_centers, size_labels)
    plt.xlabel("Matrix size")
    plt.ylabel("Average multiplication time (ns)")
    plt.title("OpenBLAS-style Matrix Multiplication Time by Matrix Size")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=160)
    plt.close()


def main() -> int:
    default_generator = default_executable_name("fg")
    default_ob = default_executable_name("ob")

    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-size matrix benchmarks (4 sizes x 10 runs), then compare ob times."
        )
    )
    parser.add_argument(
        "--runs-per-size",
        type=int,
        default=RUNS_PER_SIZE,
        help="Benchmark repetitions for each matrix size (default: 10).",
    )
    parser.add_argument(
        "--generator",
        default=default_generator,
        help=(
            "Generator executable name or path "
            f"(default: {default_generator})."
        ),
    )
    parser.add_argument(
        "--ob",
        default=default_ob,
        help=f"Baseline executable name or path (default: {default_ob}).",
    )
    args = parser.parse_args()

    if args.runs_per_size <= 0:
        raise ValueError("--runs-per-size must be a positive integer")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    generator_exe = resolve_exe(args.generator)   #fg
    ob_exe = resolve_exe(args.ob)   #ob

    ensure_executable_exists(generator_exe)
    ensure_executable_exists(ob_exe)

    rows: list[dict[str, int]] = []

    try:
        total = len(SIZES) * args.runs_per_size
        done = 0
        for size in SIZES:
            for rep in range(1, args.runs_per_size + 1):
                rows.append(run_cycle(size, rep, generator_exe, ob_exe))
                done += 1
                print(f"finish {done}/{total} (size={size}, rep={rep})", flush=True)

        averages = compute_averages(rows)
        save_rows_csv(rows)
        save_averages_csv(averages)
        draw_grouped_bar_chart(averages)

        print(f"Saved {len(rows)} benchmark rows to {RESULT_CSV}")
        print(f"Saved {len(averages)} averages to {AVERAGE_CSV}")
        print(f"Saved bar chart to {PLOT_FILE}")
    finally:
        delete_input_dir()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())