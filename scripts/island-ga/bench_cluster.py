import subprocess
import time
import csv
import argparse
import sys

# Base MPI arguments provided by the user
BASE_MPI_ARGS = [
    "mpirun",
    "-x", "DISPLAY=",
    "--mca", "plm_rsh_args", "-x",
    "--mca", "btl_tcp_if_include", "10.227.150.0/24"
]

# Define the 3 benchmark configurations (9 processes total)
configurations = [
    {
        "name": "1_Machine_Best",
        "hosts": "master:0,worker1:9,worker3:0",
        "oversubscribe": True # Needed because 9 processes > 8 cores on worker1
    },
    {
        "name": "2_Machines_TwoBest",
        "hosts": "master:4,worker1:5,worker3:0",
        "oversubscribe": False
    },
    {
        "name": "3_Machines_All",
        "hosts": "master:3,worker1:4,worker3:2",
        "oversubscribe": False
    }
]

def run_benchmark(executable, config_path, repeats, csv_filename):
    results = []

    for config in configurations:
        print(f"\n========================================")
        print(f"Running Configuration: {config['name']}")
        print(f"Hosts: {config['hosts']}")
        print(f"========================================\n")

        total_time = 0.0

        # Build the command dynamically
        cmd = BASE_MPI_ARGS.copy()
        
        if config["oversubscribe"]:
            cmd.append("--oversubscribe")
        
        cmd.extend(["--host", config["hosts"]])
        
        # Append executable and the config path as the last argument
        cmd.append(executable)
        cmd.append(config_path)

        for run in range(1, repeats + 1):
            print(f"  -> Iteration {run}/{repeats}...", end="", flush=True)
            
            start_time = time.time()
            
            # Execute the MPI process
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as e:
                print(f" [FAILED]")
                print(f"MPI Error: {e.stderr.decode('utf-8')}")
                sys.exit(1) # Stop the script entirely if MPI fails

            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed
            
            print(f" {elapsed:.2f} seconds")

        avg_time = total_time / repeats
        print(f"\n>>> Average Time for {config['name']}: {avg_time:.2f} seconds")
        
        results.append({
            "Configuration": config["name"],
            "Hosts Mapping": config["hosts"],
            "Average Time (s)": round(avg_time, 2)
        })

    # Write to CSV
    print(f"\nWriting results to {csv_filename}...")
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ["Configuration", "Hosts Mapping", "Average Time (s)"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print("Done! You can now plot these results.")

if __name__ == "__main__":
    # Setup Argument Parsing
    parser = argparse.ArgumentParser(
        description="Benchmark Hybrid MPI+OpenMP Island Model GA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mandatory positional argument
    parser.add_argument(
        "config", 
        type=str, 
        help="Path to the YAML configuration file for the GA."
    )
    
    # Optional arguments
    parser.add_argument(
        "--repeats", 
        type=int, 
        default=3, 
        help="Number of times to repeat each test configuration."
    )
    parser.add_argument(
        "--exec", 
        type=str, 
        default="./island_example", 
        dest="executable",
        help="Path to the GA executable."
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="benchmark_results.csv", 
        dest="output_csv",
        help="Name of the output CSV file."
    )

    args = parser.parse_args()

    # Start the benchmark
    run_benchmark(args.executable, args.config, args.repeats, args.output_csv)
