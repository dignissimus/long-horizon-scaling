import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(run_name: str):
    results_dir = f"experiments/results/2026-06-26-error-category-game-score-experiment/{run_name}"
    csv_path = os.path.join(results_dir, "parsed_trajectory_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Ensure parse_ablation_results.py has been run first.")
        return
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract the final row for each game
    final_df = df.sort_values("Step").groupby(["Config", "Seed"]).last().reset_index()
    
    # The Cumulative_ALE_Spatial column is now a mathematically rigorous rolling average of the error rate
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Plot average ALE rate by config
    sns.barplot(data=final_df, x="Config", y="Cumulative_ALE_Spatial", errorbar=("ci", 80), capsize=0.1)
    
    plt.title("Average ALE Rate by Configuration (80% CI)")
    plt.xlabel("Configuration (Mechanism Combination)")
    plt.ylabel("Avg ALE Rate Per Step (Lower is Better)")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    out_path = os.path.join(results_dir, "barplot_ale_rate_by_config.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Plotted {out_path} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="The exact run name (including timestamp) to analyze")
    args = parser.parse_args()
    
    main(args.run_name)
