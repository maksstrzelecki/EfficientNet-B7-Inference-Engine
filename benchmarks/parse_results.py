"""Summarize every .txt benchmark log in benchmarks/results/."""

import csv
import math
import re
from pathlib import Path
from statistics import mean, median

# Always use the directory where this script is saved.
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORTS = HERE / "reports"


def concurrency_from_name(filename):
    match = re.search(r"_c(\d+)(?:_|\.)", filename)
    return int(match.group(1)) if match else 0


def read_log_text(file):
    """Read logs written as UTF-8 or UTF-16 (PowerShell on Windows)."""
    data = file.read_bytes()

    # Windows PowerShell 5.1 often writes redirected output as UTF-16.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")

    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Fallback for older Windows console output saved in a local code page.
        return data.decode("cp1250")


def read_results(file):
    """Read individual requests and throughput from one .txt file."""
    requests = []
    throughput = None

    for line in read_log_text(file).splitlines():
        if line.startswith("Request "):
            number = re.search(r"Request (\d+):", line)
            timings = re.findall(r"(\w+)=([\d.]+) ms", line)
            if number and timings:
                request = {"request": int(number.group(1))}
                for metric, value in timings:
                    request[metric] = float(value)
                requests.append(request)

        elif "requests in" in line:
            match = re.search(r"\(([\d.]+) req/s\)", line)
            if match:
                throughput = float(match.group(1))

    return requests, throughput


def calculate_stats(values):
    """P95 is the value at rank ceil(95% of N) in sorted measurements."""
    values = sorted(values)
    p95 = values[math.ceil(0.95 * len(values)) - 1]
    return round(mean(values), 2), round(median(values), 2), p95


def main():
    REPORTS.mkdir(exist_ok=True)
    files = sorted(RESULTS.glob("*.txt"), key=lambda f: concurrency_from_name(f.name))
    if not files:
        raise SystemExit(f"No input .txt files found in {RESULTS}")

    raw_data = []
    summary = []
    overview = []

    for file in files:
        requests, throughput = read_results(file)
        if not requests:
            print(f"Skipped {file.name}: no 'Request ...' measurements found")
            continue

        concurrency = concurrency_from_name(file.name)
        metrics = sorted({key for request in requests for key in request if key != "request"})
        current_stats = {}

        for request in requests:
            raw_data.append({"file": file.name, "concurrency": concurrency, **request})

        for metric in metrics:
            values = [request[metric] for request in requests if metric in request]
            avg, med, p95 = calculate_stats(values)
            current_stats[metric] = (avg, med, p95)
            summary.append([file.name, concurrency, metric, len(values), avg, med, p95])

        latency = current_stats.get("client_latency", ("N/A", "N/A", "N/A"))
        overview.append([concurrency, file.name, len(requests), latency[0], latency[2],
                         throughput if throughput is not None else "N/A"])
        print(f"Read {file.name}: {len(requests)} requests")

    if not overview:
        raise SystemExit("No valid request data was found. Check your log format.")

    # CSV with every request, useful for later plotting or debugging.
    columns = sorted({key for row in raw_data for key in row} -
                     {"file", "concurrency", "request"})
    with (REPORTS / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "concurrency", "request", *columns])
        writer.writeheader()
        writer.writerows(raw_data)

    # CSV with statistics for each metric at each concurrency level.
    with (REPORTS / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "concurrency", "metric", "count", "mean_ms", "median_ms", "p95_ms"])
        writer.writerows(summary)

    # Markdown report that VS Code can display as formatted tables.
    lines = ["# Benchmark report", "", "## Concurrency comparison", "",
             "| Concurrency | File | Requests | Mean latency (ms) | P95 latency (ms) | Throughput (req/s) |",
             "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in overview:
        lines.append("| " + " | ".join(map(str, row)) + " |")

    lines += ["", "## All metrics", "",
              "| File | Concurrency | Metric | N | Mean (ms) | Median (ms) | P95 (ms) |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| " + " | ".join(map(str, row)) + " |")

    lines += ["", "P95 is calculated using the nearest-rank method.", ""]
    (REPORTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nReports saved in: {REPORTS}")
    print(" - summary.md\n - summary.csv\n - raw_measurements.csv")


if __name__ == "__main__":
    main()
