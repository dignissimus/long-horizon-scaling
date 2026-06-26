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

def calculate_ale(completion: str, ground_truth: list[str]) -> int:
    """
    Returns the number of Accepted Local Errors (hallucinations + forgotten items).
    """
    completion_lines = [clean_item_name(line) for line in completion.split('\n') if line.strip()]
    
    # If the model successfully utilized the explicit escape hatch, it thinks the list is empty.
    if len(completion_lines) == 1 and completion_lines[0] == "nothing":
        completion_lines = []
        
    cleaned_gt = [clean_item_name(gt) for gt in ground_truth]
    
    errors = 0
    # Check for forgotten items (False Negatives)
    for gt_item in cleaned_gt:
        if not any(gt_item in comp for comp in completion_lines):
            errors += 1
            
    # Check for hallucinated items (False Positives)
    for comp in completion_lines:
        if not any(gt_item in comp for gt_item in cleaned_gt):
            errors += 1
            
    return errors

def parse_logs(log_dir, eval_prefix):
    all_data = []
    
    # Inspect saves logs as JSON files
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
            
        # We only care about our specific experiment
        eval_name = data.get("eval", {}).get("task", "")
        if not eval_name.startswith(f"{eval_prefix}_"):
            continue
            
        config_name = eval_name.replace(f"{eval_prefix}_", "")
        
        for sample in data.get("samples", []):
            seed = sample.get("metadata", {}).get("seed", "unknown")
            step_logs = sample.get("metadata", {}).get("trajectory_telemetry", [])
            
            cumulative_ale_spatial = 0
            cumulative_ale_recipe = 0
            cumulative_interface = 0
            cumulative_liveness = 0
            cumulative_info_seek = 0
            has_read_cookbook = 0
            
            action_history = []
            
            for step_data in step_logs:
                step_idx = step_data.get("step", 0)
                score = step_data.get("reward", 0.0)
                probe_results = step_data.get("probe_results", [])
                action_sent = step_data.get("action_sent", "")
                valid_actions = step_data.get("valid_actions", [])
                
                step_ale_spatial = 0
                step_ale_recipe = 0
                drift_completion = ""
                drift_game_goal = ""
                
                for pr in probe_results:
                    if pr.get("probe") == "ale":
                        comp = pr.get("completion", "")
                        gt = pr.get("metadata", {}).get("ground_truth", [])
                        question_id = pr.get("question_id", pr.get("id", ""))
                        # Fallback for question_id parsing
                        # Wait, probe_results dictionaries don't always have 'question_id', they might be stored differently. 
                        # Inspect saves the id from ProbeQuestion into the probe results? 
                        # Let's check how we can differentiate. The prompt text or the gt?
                        # It's better to just check if "recipe" is in the prompt or ID.
                        # Wait, the id is usually returned in the probe results by our harness!
                        # The solver.py adds: "probe": probe.name, "id": question.id
                        q_id = pr.get("id", "")
                        
                        errs = calculate_ale(comp, gt)
                        if q_id.startswith("recipe"):
                            step_ale_recipe += errs
                        else:
                            step_ale_spatial += errs
                            
                    elif pr.get("probe") == "drift":
                        drift_completion = pr.get("completion", "")
                        drift_game_goal = pr.get("metadata", {}).get("game_goal", "")
                        
                cumulative_ale_spatial += step_ale_spatial
                cumulative_ale_recipe += step_ale_recipe
                
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
                    if prev_action.startswith("go ") or prev_action.startswith("open door to"):
                        if not any(curr_action.startswith(prefix) for prefix in ["look", "examine", "inventory", "read"]):
                            step_info_seek = 1
                cumulative_info_seek += step_info_seek
                
                all_data.append({
                    "Config": config_name,
                    "Seed": seed,
                    "Step": step_idx,
                    "Score": score,
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
                
    return pd.DataFrame(all_data)

def plot_results(df: pd.DataFrame, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    
    if df.empty:
        print("No data found to plot. Run the experiment first!")
        return

    sns.set_theme(style="whitegrid")
    
    # Plot 1: Cumulative Spatial ALE Errors over Steps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Cumulative_ALE_Spatial", hue="Config", errorbar="ci")
    plt.title("Cumulative Spatial ALE (Environment Hallucinations) over Horizon")
    plt.xlabel("Step Index (Proxy for Horizon)")
    plt.ylabel("Cumulative Spatial ALE")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cumulative_ale_spatial_vs_step.png"), dpi=300)
    plt.close()

    # Plot 1b: Cumulative Recipe ALE Errors over Steps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Cumulative_ALE_Recipe", hue="Config", errorbar="ci")
    plt.title("Cumulative Recipe ALE (Cookbook Hallucinations) over Horizon")
    plt.xlabel("Step Index (Proxy for Horizon)")
    plt.ylabel("Cumulative Recipe ALE")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cumulative_ale_recipe_vs_step.png"), dpi=300)
    plt.close()
    
    # Plot 2: Task Score over Steps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Score", hue="Config", errorbar="ci")
    plt.title("Task Score Progress over Horizon")
    plt.xlabel("Step Index (Proxy for Horizon)")
    plt.ylabel("Normalized Game Score")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "score_vs_step.png"), dpi=300)
    plt.close()
    
    # Plot 3: Cumulative Info Seeking Errors over Steps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Cumulative_Info_Seek", hue="Config", errorbar="ci")
    plt.title("Cumulative Information-Seeking Errors over Horizon")
    plt.xlabel("Step Index (Proxy for Horizon)")
    plt.ylabel("Cumulative Info-Seeking Errors (Unverified Room Entries)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cumulative_infoseek_vs_step.png"), dpi=300)
    plt.close()
    
    # Plot 4: Cookbook Read Progress
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Has_Read_Cookbook", hue="Config", errorbar="ci")
    plt.title("Proportion of Agents that have Read the Cookbook")
    plt.xlabel("Step Index (Proxy for Horizon)")
    plt.ylabel("Has Read Cookbook (1=Yes, 0=No)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cookbook_read_vs_step.png"), dpi=300)
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
