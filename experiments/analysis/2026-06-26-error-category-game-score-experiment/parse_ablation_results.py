import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOG_DIR = "logs/error_probes"
RESULTS_DIR = "experiments/results/error_probes"

def clean_item_name(item: str) -> str:
    """Removes 'a', 'an', 'the' and strips whitespace for fuzzy matching."""
    item = item.lower().strip()
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
            
            cumulative_ale = 0
            cumulative_interface = 0
            cumulative_liveness = 0
            
            action_history = []
            
            for step_data in step_logs:
                step_idx = step_data.get("step", 0)
                score = step_data.get("reward", 0.0)
                probe_results = step_data.get("probe_results", [])
                action_sent = step_data.get("action_sent", "")
                valid_actions = step_data.get("valid_actions", [])
                
                step_ale = 0
                drift_completion = ""
                drift_game_goal = ""
                
                for pr in probe_results:
                    if pr.get("probe") == "ale":
                        comp = pr.get("completion", "")
                        gt = pr.get("metadata", {}).get("ground_truth", [])
                        step_ale += calculate_ale(comp, gt)
                    elif pr.get("probe") == "drift":
                        drift_completion = pr.get("completion", "")
                        drift_game_goal = pr.get("metadata", {}).get("game_goal", "")
                        
                cumulative_ale += step_ale
                
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
                
                all_data.append({
                    "Config": config_name,
                    "Seed": seed,
                    "Step": step_idx,
                    "Score": score,
                    "Step_ALE": step_ale,
                    "Cumulative_ALE": cumulative_ale,
                    "Step_Interface": step_interface,
                    "Cumulative_Interface": cumulative_interface,
                    "Step_Liveness": step_liveness,
                    "Cumulative_Liveness": cumulative_liveness,
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
    
    # Plot 1: Cumulative ALE Errors over Steps
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Cumulative_ALE", hue="Config", errorbar="ci")
    plt.title("Cumulative Accepted Local Errors (ALE) over Horizon")
    plt.xlabel("Step Index (Proxy for Horizon)")
    plt.ylabel("Cumulative ALE (Hallucinations & Forgotten Items)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cumulative_ale_vs_step.png"), dpi=300)
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
