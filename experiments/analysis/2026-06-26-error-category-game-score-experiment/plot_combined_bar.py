import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

MECHANISM_NAMES = {
    "m3": "Given Inventory State",
    "m6": "Planning",
    "m7": "Given World State"
}

def get_label(config: str) -> str:
    parts = config.split('_')
    labels = []
    if "m6" in parts:
        labels.append(MECHANISM_NAMES["m6"])
    if "m3" in parts:
        labels.append(MECHANISM_NAMES["m3"])
    if "m7" in parts:
        labels.append(MECHANISM_NAMES["m7"])
        
    # Only label as Baseline if no other experimental mechanisms are active
    if len(labels) == 0:
        labels = ["Baseline"]
        
    return "\n".join(labels)


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
    
    # Map the Config column dynamically without hardcoding
    final_df["Mechanisms enabled"] = final_df["Config"].apply(get_label)
    
    # Convert fractional error to a percentage
    final_df["Knowledge_Error_Pct"] = final_df["Cumulative_ALE_Spatial"] * 100
    
    # Sort X-axis dynamically by complexity (number of mechanisms), then alphabetically
    present_configs = final_df["Mechanisms enabled"].unique()
    order = sorted(present_configs, key=lambda x: (len(x.split('\n')), x))
    
    sns.set_theme(style="whitegrid")
    
    # Create a 2x1 vertical subplot layout in a taller landscape mode
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    
    # ----------------------------------------------------
    # Plot 1 (Top): % Incorrect Knowledge of State (ALE)
    # ----------------------------------------------------
    sns.barplot(
        data=final_df, x="Mechanisms enabled", y="Knowledge_Error_Pct", 
        order=order, errorbar=("ci", 80), capsize=0.1, ax=axes[0], color="salmon"
    )
    axes[0].set_title("% Items Forgotten", fontsize=16, pad=15)
    axes[0].set_xlabel("")  # Hide x-label title to prevent clutter, but keep the tick labels
    axes[0].set_ylabel("% Items Forgotten (Lower is Better)", fontsize=12)
    axes[0].yaxis.set_major_formatter(PercentFormatter(100))
    axes[0].tick_params(axis='x', rotation=45)
    
    # Align labels
    for label in axes[0].get_xticklabels():
        label.set_ha('right')
    
    # ----------------------------------------------------
    # Plot 2 (Bottom): Final Game Score
    # ----------------------------------------------------
    sns.barplot(
        data=final_df, x="Mechanisms enabled", y="Score", 
        order=order, errorbar=("ci", 80), capsize=0.1, ax=axes[1], color="skyblue"
    )
    axes[1].set_title("Final Game Score", fontsize=16, pad=15)
    axes[1].set_xlabel("Mechanisms enabled", fontsize=14)
    axes[1].set_ylabel("Game Score (Higher is Better)", fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Align labels
    for label in axes[1].get_xticklabels():
        label.set_ha('right')
        
    # Add footnote at the bottom
    plt.figtext(0.5, 0.01, "* Note: All combinations include the Baseline mechanism (Given Valid Actions)", 
                ha='center', fontsize=12, fontstyle='italic')
        
    # Adjust layout and leave room at the bottom for the figtext
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    out_path = os.path.join(results_dir, "combined_barplot_results.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Plotted {out_path} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="The exact run name (including timestamp) to analyze")
    args = parser.parse_args()
    
    main(args.run_name)
