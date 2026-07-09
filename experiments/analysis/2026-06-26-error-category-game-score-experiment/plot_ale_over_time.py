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
        
    # Use " + " instead of newlines for a cleaner legend in lineplots
    return " + ".join(labels)

def main(run_name: str):
    results_dir = f"experiments/results/2026-06-26-error-category-game-score-experiment/{run_name}"
    csv_path = os.path.join(results_dir, "parsed_trajectory_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Ensure parse_ablation_results.py has been run first.")
        return
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Map the Config column dynamically without hardcoding
    df["Mechanisms enabled"] = df["Config"].apply(get_label)
    
    # Convert fractional error to a percentage
    df["Knowledge_Error_Pct"] = df["Cumulative_ALE_Spatial"] * 100
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 8))
    
    # Create the lineplot averaging over seeds
    sns.lineplot(
        data=df, 
        x="Step", 
        y="Knowledge_Error_Pct", 
        hue="Mechanisms enabled",
        errorbar=None, # Mean over seeds
        linewidth=2
    )
    
    plt.title("% Items Forgotten", fontsize=16, pad=15)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("% Items Forgotten (Lower is Better)", fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
    
    plt.legend(title="Mechanisms enabled", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add footnote at the bottom
    plt.figtext(0.5, 0.01, "* Note: All combinations include the Baseline mechanism (Given Valid Actions)", 
                ha='center', fontsize=12, fontstyle='italic')
        
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    out_path = os.path.join(results_dir, "ale_over_time.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.8)
    plt.close()
    
    print(f"Plotted {out_path} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="The exact run name (including timestamp) to analyze")
    args = parser.parse_args()
    
    main(args.run_name)
