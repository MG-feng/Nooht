"""记录各后端 selective scan 基线性能。

用法: python scripts/bench_scan.py [--batch 2 --dim 256 --length 1024 --state 16 --iters 5]
输出: benchmarks/scan_baseline.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from nooht.backends import list_backends


def bench_one(backend, B, D, L, N, iters):
    backend.setup()
    device = backend.default_device()
    x = torch.randn(B, D, L).to(device)
    dt = torch.randn(B, D, L).to(device)
    A = (torch.rand(D, N) + 0.5).to(device)
    Bm = torch.randn(B, N, L).to(device)
    Cm = torch.randn(B, N, L).to(device)
    Dm = torch.randn(D).to(device)

    for _ in range(3):  # warmup
        backend.selective_scan(x, dt, A, Bm, Cm, Dm, delta_softplus=True)
    backend.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        backend.selective_scan(x, dt, A, Bm, Cm, Dm, delta_softplus=True)
    backend.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--state", type=int, default=16)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", default="benchmarks/scan_baseline.csv")
    args = p.parse_args()

    rows = []
    for backend in list_backends(only_available=True):
        try:
            ms = bench_one(backend, args.batch, args.dim, args.length,
                           args.state, args.iters)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {backend.name}: {exc}")
            continue
        print(f"{backend.name:>6}: {ms:8.2f} ms/iter")
        rows.append({"backend": backend.name,
                     "device": str(backend.default_device()),
                     "B": args.batch, "D": args.dim, "L": args.length,
                     "N": args.state, "ms_per_iter": f"{ms:.3f}"})

    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"written: {args.out}")
    else:
        raise SystemExit("no backend produced a baseline row")


if __name__ == "__main__":
    main()