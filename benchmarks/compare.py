import json
import os
from .baseline_train import train_baseline
from .nrt_train import run_nrt_benchmark

def run_comparison(total_steps=500, checkpoint_interval=50, io_delay=2.0):
    print("="*60)
    print("NOOHT NRT BENCHMARK: BASELINE vs NRT")
    print("="*60)
    
    baseline = train_baseline(total_steps, checkpoint_interval, io_delay)
    nrt = run_nrt_benchmark(total_steps, checkpoint_interval, io_delay)
    
    b_idle = sum(baseline.gpu_idle_times)
    n_idle = sum(nrt.gpu_idle_times)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} | {'Baseline':<15} | {'NRT':<15} | {'Delta'}")
    print("-"*60)
    print(f"{'Total Time (s)':<20} | {baseline.total_time:<15.2f} | {nrt.total_time:<15.2f} | {baseline.total_time - nrt.total_time:+.2f}s")
    print(f"{'Checkpoints':<20} | {baseline.checkpoints_saved:<15} | {nrt.checkpoints_saved:<15} | -")
    print("="*60)
    
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/report.json", "w") as f:
        json.dump({"baseline_total": baseline.total_time, "nrt_total": nrt.total_time, "idle_eliminated": b_idle - n_idle}, f)
    print("Report saved to benchmarks/results/report.json")

if __name__ == "__main__":
    run_comparison()
