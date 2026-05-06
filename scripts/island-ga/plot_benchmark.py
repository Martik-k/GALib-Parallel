import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def main():
    # 1. Setup Argument Parsing
    parser = argparse.ArgumentParser(
        description="Generate and save a bar chart from the Hybrid GA benchmark CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mandatory positional argument: CSV Input
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the benchmark CSV file (e.g., benchmark_results.csv)."
    )

    # Mandatory positional argument: Image Output
    parser.add_argument(
        "out_path",
        type=str,
        help="Path to save the generated plot image (e.g., results_plot.png)."
    )

    args = parser.parse_args()

    # 2. Validate File Exists
    if not os.path.exists(args.csv_path):
        print(f"Error: The input file '{args.csv_path}' does not exist.")
        sys.exit(1)

    print(f"Loading data from {args.csv_path}...")

    # 3. Read the CSV file
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(1)

    # Verify expected columns exist
    expected_cols = ["Configuration", "Average Time (s)"]
    for col in expected_cols:
        if col not in df.columns:
            print(f"Error: CSV is missing the required column: '{col}'")
            sys.exit(1)

    # 4. Map configuration names to explicitly show the cores on EACH machine
    # Worker1 = 8, Master = 6, Worker3 = 4
    name_mapping = {
        "1_Machine_Best": "1 Machine\n(8 cores)",
        "2_Machines_TwoBest": "2 Machines\n(8 + 6 cores)",
        "3_Machines_All": "3 Machines\n(8 + 6 + 4 cores)"
    }

    # Apply the mapping to the dataframe
    df["Configuration"] = df["Configuration"].map(lambda x: name_mapping.get(x, x))

    # 5. Set up the visual style (Consistent Matplotlib/Seaborn theme)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create the bar plot
    ax = sns.barplot(
        x="Configuration",
        y="Average Time (s)",
        data=df,
        palette="viridis",
        hue="Configuration",
        legend=False
    )

    # Add the exact time values on top of each bar
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), '.2f') + 's',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 10),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )

    # 6. Formatting the plot
    plt.title('Hybrid Island GA Cluster Benchmark\n(Fixed Workload: 9 Islands)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Cluster Configuration', fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel('Average Execution Time (Seconds)', fontsize=13, fontweight='bold', labelpad=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)

    # Adjust layout so labels don't get cut off
    plt.tight_layout()

    # 7. Save the plot strictly (No plt.show())
    plt.savefig(args.out_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {os.path.abspath(args.out_path)}")

if __name__ == "__main__":
    main()