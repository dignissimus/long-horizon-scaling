import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOG_DIR = "logs/error_probes"
RESULTS_DIR = "experiments/results/error_probes"

import string

def clean_item_name(item: str) -> str:
    """Removes 'a', 'an', 'the', strips punctuation, and strips whitespace for fuzzy matching."""
    item = item.lower().strip()
    
    # Strip punctuation
    item = item.translate(str.maketrans('', '', string.punctuation))
    
    prefixes = ['a ', 'an ', 'the ', 'some ']
    for p in prefixes:
        if item.startswith(p):
            item = item[len(p):]
    return item.strip()

def calculate_ale(completion: str, ground_truth: list[str]) -> tuple[int, int]:
    """
    Returns a tuple of (forgotten_items, total_items).
    Returns 0.0 if the environment has nothing.
    Hallucinations are ignored per user request.
    Filters out common structural surfaces and containers since the agent was told to ignore them.
    """
    completion_lines = [clean_item_name(line) for line in completion.split('\n') if line.strip()]
    
    # If the model successfully utilized the explicit escape hatch, it thinks the list is empty.
    if len(completion_lines) == 1 and completion_lines[0] == "nothing":
        completion_lines = []
        
    cleaned_gt = [clean_item_name(gt) for gt in ground_truth]
    
    if len(cleaned_gt) == 0:
        return (0, 0)
    
    forgotten = 0
    # Check for forgotten items (False Negatives)
    for gt_item in cleaned_gt:
        if not any(gt_item in comp for comp in completion_lines):
            forgotten += 1
            
    # Check for hallucinated items (False Positives)
    # for comp in completion_lines:
    #     if not any(gt_item in comp for gt_item in cleaned_gt):
    #         errors += 1
            
    return (forgotten, len(cleaned_gt))

from inspect_ai.log import read_eval_log

def parse_logs(log_dir, eval_prefix):
    all_data = []
    
    # Inspect saves logs as .eval files (which are JSON/MessagePack formatted)
    log_files = glob.glob(os.path.join(log_dir, "*.eval"))
    # Also support .json in case they were manually renamed
    log_files.extend(glob.glob(os.path.join(log_dir, "*.json")))
    
    for log_file in log_files:
        try:
            if log_file.endswith(".eval"):
                data = read_eval_log(log_file)
                eval_name = data.eval.task
                samples = data.samples
            else:
                with open(log_file, 'r') as f:
                    raw_data = json.load(f)
                eval_name = raw_data.get("eval", {}).get("task", "")
                samples = raw_data.get("samples", [])
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
            
        # The eval_name is typically in the format: "<experiment_name>_<config_name>"
        # We can extract the config_name by finding the longest matching known config,
        # or by splitting if we strictly enforce the naming convention.
        config_name = "unknown"
        known_configs = ["baseline_m5", "m3_m5", "m7_m5", "m5_m6", "m3_m5_m6", "m7_m5_m6"]
        
        # Sort by length descending to prevent substring collisions (e.g., m5_m6 vs m3_m5_m6)
        known_configs.sort(key=len, reverse=True)
        
        for known in known_configs:
            if eval_name.endswith(f"_{known}"):
                config_name = known
                break
                
        if config_name == "unknown":
            print(f"Skipping {eval_name}: Could not match to a known configuration.")
            continue
        
        for sample in samples:
            # Handle both EvalSample objects (from read_eval_log) and dicts (from json.load)
            metadata = sample.metadata if hasattr(sample, 'metadata') else sample.get("metadata", {})
            seed = metadata.get("seed", "unknown")
            step_logs = metadata.get("trajectory_telemetry", [])
            
            cumulative_ale_spatial = 0.0
            cumulative_ale_recipe = 0.0
            
            # True rolling average trackers for ALE metrics
            cum_spatial_forgotten = 0
            cum_spatial_total = 0
            cum_recipe_forgotten = 0
            cum_recipe_total = 0
            
            cumulative_interface = 0
            cumulative_liveness = 0
            cumulative_info_seek = 0
            cumulative_score = 0.0
            has_read_cookbook = 0
            
            action_history = []
            
            for step_data in step_logs:
                step_idx = step_data.get("step", 0)
                score_delta = step_data.get("reward", 0.0)
                cumulative_score += score_delta
                probe_results = step_data.get("probe_results", [])
                action_sent = step_data.get("action_sent", "")
                valid_actions = step_data.get("valid_actions", [])
                
                step_spatial_forgotten = 0
                step_spatial_total = 0
                step_recipe_forgotten = 0
                step_recipe_total = 0
                
                drift_completion = ""
                drift_game_goal = ""
                
                for pr in probe_results:
                    if pr.get("probe") == "ale":
                        comp = pr["completion"]
                        gt = pr["metadata"]["ground_truth"]
                        
                        # Strict control flow for identifying the question ID
                        if "question_id" in pr:
                            q_id = pr["question_id"]
                        elif "id" in pr:
                            q_id = pr["id"]
                        else:
                            raise KeyError("ALE probe result is missing both 'question_id' and 'id'")
                        
                        forgotten, total = calculate_ale(comp, gt)
                        if q_id.startswith("recipe"):
                            step_recipe_forgotten += forgotten
                            step_recipe_total += total
                        else:
                            step_spatial_forgotten += forgotten
                            step_spatial_total += total
                            
                    elif pr.get("probe") == "drift":
                        drift_completion = pr.get("completion", "")
                        drift_game_goal = pr.get("metadata", {}).get("game_goal", "")
                        
                # Update true rolling totals
                cum_spatial_forgotten += step_spatial_forgotten
                cum_spatial_total += step_spatial_total
                cum_recipe_forgotten += step_recipe_forgotten
                cum_recipe_total += step_recipe_total
                
                # Calculate normalized fractional rates
                step_ale_spatial = step_spatial_forgotten / max(1, step_spatial_total)
                step_ale_recipe = step_recipe_forgotten / max(1, step_recipe_total)
                cumulative_ale_spatial = cum_spatial_forgotten / max(1, cum_spatial_total)
                cumulative_ale_recipe = cum_recipe_forgotten / max(1, cum_recipe_total)
                
                # 2. Interface Errors
                step_interface = 0
                if action_sent and valid_actions and action_sent not in valid_actions:
                    step_interface = 1
                cumulative_interface += step_interface
                
                # 3. Liveness Errors (State-Action Attractor proxy)
                action_history.append(action_sent)
                step_liveness = 0
                if len(action_history) >= 4 and len(set(action_history[-4:])) == 1:
                    # Repeating the exact same action 4 times in a row
                    step_liveness = 1
                cumulative_liveness += step_liveness
                
                # 4. Information Seeking Errors
                if "read cookbook" in action_sent:
                    has_read_cookbook = 1
                    
                step_info_seek = 0
                if len(action_history) >= 2:
                    prev_action = action_history[-2]
                    curr_action = action_history[-1]
                    # Note: We previously penalized for not looking after moving, but TextWorld 
                    # auto-looks on room transitions. Thus, we only penalize opening closed containers 
                    # blindly without examining their contents, or similar blind interactions.
                    if prev_action.startswith("open ") and not prev_action.startswith("open door"):
                        if not curr_action.startswith("look") and not curr_action.startswith("examine") and not curr_action.startswith("take"):
                            step_info_seek = 1
                cumulative_info_seek += step_info_seek
                
                all_data.append({
                    "Config": config_name,
                    "Seed": seed,
                    "Step": step_idx,
                    "Score": cumulative_score,
                    "Cum_Spatial_Total": cum_spatial_total,
                    "Step_ALE_Spatial": step_ale_spatial,
                    "Cumulative_ALE_Spatial": cumulative_ale_spatial,
                    "Step_ALE_Recipe": step_ale_recipe,
                    "Cumulative_ALE_Recipe": cumulative_ale_recipe,
                    "Step_Interface": step_interface,
                    "Cumulative_Interface": cumulative_interface,
                    "Step_Liveness": step_liveness,
                    "Cumulative_Liveness": cumulative_liveness,
                    "Step_Info_Seek": step_info_seek,
                    "Cumulative_Info_Seek": cumulative_info_seek,
                    "Has_Read_Cookbook": has_read_cookbook,
                    "Drift_Completion": drift_completion,
                    "Drift_Game_Goal": drift_game_goal
                })
    df = pd.DataFrame(all_data)
    if df.empty:
        return df
        
    min_step = df["Step"].min()
    max_step = df["Step"].max()
    
    df = df.set_index(["Config", "Seed", "Step"])
    unique_runs = df.index.droplevel("Step").unique()
    
    new_index = pd.MultiIndex.from_tuples(
        [(c, s, step) for c, s in unique_runs for step in range(int(min_step), int(max_step) + 1)],
        names=["Config", "Seed", "Step"]
    )
    
    # Reindex creates NaN rows for steps where the agent had already died
    df = df.reindex(new_index)
    
    # Forward-fill Score and Cumulative_ALE_Spatial to prevent survival bias
    df["Score"] = df.groupby(level=["Config", "Seed"])["Score"].ffill()
    df["Cumulative_ALE_Spatial"] = df.groupby(level=["Config", "Seed"])["Cumulative_ALE_Spatial"].ffill()
    df["Cum_Spatial_Total"] = df.groupby(level=["Config", "Seed"])["Cum_Spatial_Total"].ffill()
    
    df = df.reset_index()
    return df
def plot_results(df: pd.DataFrame, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    
    if df.empty:
        print("No data found to plot. Run the experiment first!")
        return

    sns.set_theme(style="whitegrid")
    
    # Plot 1: Task Score over Steps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Score", hue="Config", errorbar=("ci", 80))
    plt.title("Task Score Progress over Horizon")
    plt.xlabel("Step Index (Proxy for Horizon)")
    plt.ylabel("Normalized Game Score")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "score_vs_step.png"), dpi=300)
    plt.close()

    # Plot 5: Scatter plot of Final Cumulative ALE vs Final Score
    # Get the final row for each Seed+Config
    final_df = df.sort_values("Step").groupby(["Config", "Seed"]).last().reset_index()
    final_df["Total_Final_ALE"] = final_df["Cumulative_ALE_Spatial"] + final_df["Cumulative_ALE_Recipe"]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=final_df, x="Total_Final_ALE", y="Score", hue="Config", s=100, alpha=0.7)
    plt.title("Final Cumulative ALE (Spatial + Recipe) vs Final Score")
    plt.xlabel("Total Final ALE (Hallucinations + Forgotten)")
    plt.ylabel("Final Game Score")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "scatter_final_ale_vs_score.png"), dpi=300)
    plt.close()
    
    # Plot 6: Bar plot for Final Score by Config with 80% CI
    plt.figure(figsize=(12, 6))
    sns.barplot(data=final_df, x="Config", y="Score", errorbar=("ci", 80), capsize=0.1)
    plt.title("Final Game Score by Configuration (80% CI)")
    plt.xlabel("Configuration (Mechanism Combination)")
    plt.ylabel("Final Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "barplot_final_score_by_config.png"), dpi=300)
    plt.close()
    
    print(f"Data analysis complete. Plots saved to {results_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="The exact run name (including timestamp) to analyze")
    args = parser.parse_args()
    
    log_dir = f"logs/2026-06-26-error-category-game-score-experiment/{args.run_name}"
    results_dir = f"experiments/results/2026-06-26-error-category-game-score-experiment/{args.run_name}"
    
    print(f"Parsing inspect logs from {log_dir}...")
    # The eval_prefix should just match whatever is in eval.name. 
    # In the execution script, we set experiment_name=args.run_name (without timestamp)
    # Wait, in the execution script we set experiment_name=args.run_name, but the folder is final_run_name!
    # Let's extract the base prefix by stripping the trailing timestamp if possible, or just pass args.run_name.
    # Actually, the user passes the full folder name (with timestamp) as --run_name to the analysis script.
    # The eval task name in the json will just be the original base `args.run_name` used during generation.
    # We can just match any `error_analysis` or anything, since we're already scoped to the timestamped folder!
    df = parse_logs(log_dir, "") 
    
    if not df.empty:
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "parsed_trajectory_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved tabular data to {csv_path}")
        
        print("Generating plots...")
        plot_results(df, results_dir)
    else:
        print(f"No valid logs found in the '{log_dir}/' directory.")
