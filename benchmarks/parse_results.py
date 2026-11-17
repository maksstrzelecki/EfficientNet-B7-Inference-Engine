import re
import sys
import statistics

filename = sys.argv[1]

with open(filename, "rb") as f:
    raw = f.read()

if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
    text = raw.decode("utf-16")
else:
    text = raw.decode("utf-8-sig")
    text = text.replace("\x00", "")

metrics = [
    "preprocessing_time",
    "inference_time",
    "queue_time",
    "client_latency",
]

for metric in metrics:
    pattern = rf"\b{metric}\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    values = [
        float(v)
        for v in re.findall(pattern, text)
    ]

    if not values:
        print(f"{metric}: missing data")
        continue

    values.sort()
    p95_index = max(0, int(0.95 * len(values) + 0.999999) - 1)

    print(f"\n{metric}")
    print(f"  number of meassurments: {len(values)}")
    print(f"  mean: {statistics.mean(values):.2f} ms")
    print(f"  median: {statistics.median(values):.2f} ms")
    print(f"  P95: {values[p95_index]:.2f} ms")

print("\nBenchmark summary:")
for line in text.splitlines()[-5:]:
    print(line)
