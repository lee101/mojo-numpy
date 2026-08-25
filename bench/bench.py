from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojonumpy as mnp  # noqa: E402


def best_time(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def gpu_memory_free_mib():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return min(int(line.strip()) for line in output.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def cases(gpu_free_mib):
    rng = np.random.default_rng(2026)

    a = rng.normal(size=5_000_000)
    b = rng.normal(size=5_000_000)
    yield "add (5M)", lambda: mnp.add(a, b), lambda: np.add(a, b)

    x = rng.normal(size=2_000_000)
    yield "tanh (2M)", lambda: mnp.tanh(x), lambda: np.tanh(x)

    r = rng.normal(size=5_000_000)
    yield "sum (5M)", lambda: mnp.sum(r), lambda: np.sum(r)

    s = rng.normal(size=1_000_000)
    yield "sort (1M)", lambda: mnp.sort(s), lambda: np.sort(s)

    left = rng.normal(size=(256, 256))
    right = rng.normal(size=(256, 256))
    yield "matmul (256x256)", lambda: mnp.matmul(left, right), lambda: np.matmul(left, right)

    if gpu_free_mib >= 4000:
        gpu_left = rng.normal(size=(1024, 1024))
        gpu_right = rng.normal(size=(1024, 1024))
        yield (
            "matmul GPU (1024x1024)",
            lambda: mnp.matmul(gpu_left, gpu_right, device="gpu"),
            lambda: np.matmul(gpu_left, gpu_right),
        )

    system = rng.normal(size=(256, 256))
    system += np.eye(256) * 20
    rhs = rng.normal(size=256)
    yield "solve (256x256)", lambda: mnp.linalg.solve(system, rhs), lambda: np.linalg.solve(system, rhs)

    mojo_rng = mnp.random.default_rng(8)
    numpy_rng = np.random.default_rng(8)
    yield (
        "standard_normal (2M)",
        lambda: mojo_rng.standard_normal(2_000_000),
        lambda: numpy_rng.standard_normal(2_000_000),
    )


def equivalent(name, ours, theirs):
    if "standard_normal" in name:
        return (
            abs(float(np.mean(ours))) < 0.01
            and abs(float(np.std(ours)) - 1.0) < 0.01
            and abs(float(np.mean(theirs))) < 0.01
            and abs(float(np.std(theirs)) - 1.0) < 0.01
        )
    return np.allclose(ours, theirs, rtol=2e-9, atol=2e-9, equal_nan=True)


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    gpu_free_mib = gpu_memory_free_mib()
    if gpu_free_mib < 4000:
        detail = f"only {gpu_free_mib} MiB free" if gpu_free_mib else "no GPU available"
        print(f"GPU benchmark skipped: {detail}")
    print()
    print("| case | mojo-numpy | NumPy | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_fn, numpy_fn in cases(gpu_free_mib):
        ours = mojo_fn()
        theirs = numpy_fn()
        if not equivalent(name, ours, theirs):
            raise RuntimeError(f"benchmark parity check failed for {name}")
        mojo_seconds = best_time(mojo_fn)
        numpy_seconds = best_time(numpy_fn)
        speedup = numpy_seconds / mojo_seconds
        result = (
            f"{speedup:.2f}x faster"
            if speedup >= 1
            else f"{1.0 / speedup:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{numpy_seconds * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
