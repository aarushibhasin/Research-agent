import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from benchmark.loader import load_benchmark
from benchmark.loader import extract_haystack_chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect MemoryAgentBench split memory availability.")
    p.add_argument("--split", default=os.getenv("BENCHMARK_SPLIT", "Accurate_Retrieval"))
    p.add_argument("--scan", type=int, default=200, help="How many examples to scan from the start.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_benchmark(args.split)

    scan_n = min(len(ds), max(1, args.scan))
    non_empty = []
    for i in range(scan_n):
        ex = ds[i]
        chunks = extract_haystack_chunks(ex)
        n = len(chunks)
        if n > 0:
            non_empty.append((i, n))

    print(f"Split: {args.split}")
    print(f"Scanned first {scan_n} examples")
    print(f"Non-empty haystack_sessions examples in scan: {len(non_empty)}")
    if non_empty:
        i0, n0 = non_empty[0]
        print(f"First non-empty example_index={i0} haystack_sessions_chunks={n0}")
        # Print a few more
        for i, n in non_empty[:5]:
            print(f"  - example_index={i} chunks={n}")
    else:
        print("No non-empty haystack_sessions found in this scan window.")
        print("Try increasing --scan or verify dataset cache/download.")


if __name__ == "__main__":
    main()

