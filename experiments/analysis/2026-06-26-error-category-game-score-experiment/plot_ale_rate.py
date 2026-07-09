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
    
    sns.set_theme(style="whitegrid")
    # Plot normalized ALE rate vs Score
    g = sns.lmplot(data=final_df, x="Cumulative_ALE_Spatial", y="Score", hue="Config", 
                   scatter_kws={'alpha':0.6, 's':100}, height=6, aspect=1.5, ci=80)
    g.set_axis_labels("Average ALE per Step (Error Rate)", "Final Game Score")
    g.fig.suptitle("Predictive Power of ALE: Avg ALE per Step vs Final Score", y=1.02)
    
    plt.tight_layout()
    
    out_path = os.path.join(results_dir, "lmplot_ale_rate_vs_score.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Plotted {out_path} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="The exact run name (including timestamp) to analyze")
    args = parser.parse_args()
    
    main(args.run_name)
