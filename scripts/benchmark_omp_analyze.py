import subprocess
import time
import os
import sys
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VIS_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'images')

if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

def run_benchmark():
    executable = os.path.join(PROJECT_ROOT, 'build', 'ga_example')

    config_file = os.path.join(PROJECT_ROOT, 'configs', 'bench_scaling.yaml')

    if not os.path.exists(executable):
        print(f"Executable not found at {executable}.")
        return

    thread_counts = [1, 2, 3, 4, 5, 6, 7, 8]
    num_runs = 3

    avg_times = []

    print(f"Starting RCGA Parallel Benchmark")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Config: {config_file}\n")

    for threads in thread_counts:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)

        print(f"Testing with {threads} thread(s)")

        total_time = 0.0
        for run_idx in range(num_runs):
            start_time = time.perf_counter()

            process = subprocess.Popen(
                [executable, config_file],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            for line in process.stdout:
                if "Generation" in line:
                    parts = line.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        print(f"\r  Run {run_idx + 1}/{num_runs}: Gen {parts[1]}...", end="", flush=True)

            process.wait()
            end_time = time.perf_counter()

            if process.returncode != 0:
                print(f"\nError: {process.stderr.read()}")
                return

            run_time = end_time - start_time
            total_time += run_time
            print(f"\r  Run {run_idx + 1}/{num_runs}: {run_time:.4f}s" + " " * 10)

        avg_time = total_time / num_runs
        avg_times.append(avg_time)
        print(f"Avg for {threads} threads: {avg_time:.4f}s\n")

    base_time = avg_times[0]
    speedups = [base_time / t for t in avg_times]

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(thread_counts, avg_times, marker='o', color='#3498db', linewidth=2)
    ax1.set_title("Execution Time (Lower is Better)")
    ax1.set_xlabel("Threads")
    ax1.set_ylabel("Seconds")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xticks(thread_counts)

    ax2.plot(thread_counts, speedups, marker='s', color='#2ecc71', linewidth=2, label="RCGA Speedup")
    ax2.plot(thread_counts, thread_counts, linestyle='--', color='#e74c3c', label="Linear Scaling (Ideal)")
    ax2.set_title("Parallel Speedup (Higher is Better)")
    ax2.set_xlabel("Threads")
    ax2.set_ylabel("Speedup Factor")
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_xticks(thread_counts)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(VIS_DIR, 'omp_performance_results.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Benchmark finished! Graph saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_benchmark()