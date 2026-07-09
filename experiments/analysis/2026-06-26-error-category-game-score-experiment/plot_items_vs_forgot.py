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
        
    return " + ".join(labels)

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
    
    # Calculate absolute number of items forgotten
    final_df["Cum_Spatial_Forgotten"] = final_df["Cumulative_ALE_Spatial"] * final_df["Cum_Spatial_Total"]
    
    sns.set_theme(style="whitegrid")
    
    def make_plot(log_x=False):
        plt.figure(figsize=(10, 6))
        
        # Create the scatterplot
        sns.scatterplot(
            data=final_df, 
            x="Cum_Spatial_Total", 
            y="Knowledge_Error_Pct", 
            hue="Mechanisms enabled",
            s=100,
            alpha=0.8
        )
        
        title_suffix = " (Log Scale X)" if log_x else ""
        plt.title(f"% Items Forgotten vs Total Number of Items{title_suffix}", fontsize=16, pad=15)
        plt.xlabel("Total Number of Spatial Items (Ground Truth)", fontsize=14)
        plt.ylabel("% Items Forgotten", fontsize=12)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
        
        if log_x:
            plt.xscale('log')
        
        plt.legend(title="Mechanisms enabled", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add footnote at the bottom
        plt.figtext(0.5, -0.05, "* Note: All combinations include the Baseline mechanism (Given Valid Actions)", 
                    ha='center', fontsize=12, fontstyle='italic')
            
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        filename = "scatter_items_vs_forgot_log_x.png" if log_x else "scatter_items_vs_forgot.png"
        out_path = os.path.join(results_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.8)
        plt.close()
        
        print(f"Plotted {out_path} successfully!")

    make_plot(log_x=False)
    make_plot(log_x=True)

    def make_absolute_plot(log_x=False, log_y=False):
        plt.figure(figsize=(10, 6))
        
        # Create the scatterplot
        sns.scatterplot(
            data=final_df, 
            x="Cum_Spatial_Total", 
            y="Cum_Spatial_Forgotten", 
            hue="Mechanisms enabled",
            s=100,
            alpha=0.8
        )
        
        title_suffix_parts = []
        if log_x: title_suffix_parts.append("Log X")
        if log_y: title_suffix_parts.append("Log Y")
        title_suffix = f" ({', '.join(title_suffix_parts)})" if title_suffix_parts else ""
        
        plt.title(f"Number of Items Forgotten vs Total Number of Items{title_suffix}", fontsize=16, pad=15)
        plt.xlabel("Total Number of Spatial Items (Ground Truth)", fontsize=14)
        plt.ylabel("Number of Items Forgotten", fontsize=14)
        
        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('log')
        
        plt.legend(title="Mechanisms enabled", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add footnote at the bottom
        plt.figtext(0.5, -0.05, "* Note: All combinations include the Baseline mechanism (Given Valid Actions)", 
                    ha='center', fontsize=12, fontstyle='italic')
            
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        filename = "scatter_abs_items_vs_forgot"
        if log_x: filename += "_log_x"
        if log_y: filename += "_log_y"
        filename += ".png"
        
        out_path = os.path.join(results_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.8)
        plt.close()
        
        print(f"Plotted {out_path} successfully!")

    make_absolute_plot(log_x=False, log_y=False)
    make_absolute_plot(log_x=True, log_y=False)
    make_absolute_plot(log_x=True, log_y=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="The exact run name (including timestamp) to analyze")
    args = parser.parse_args()
    
    main(args.run_name)
