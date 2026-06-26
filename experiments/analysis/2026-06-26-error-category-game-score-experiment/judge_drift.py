import os
import argparse
import pandas as pd
import asyncio
from inspect_ai.model import get_model

# System prompt for the LLM Judge
JUDGE_SYSTEM_PROMPT = """You are an expert AI evaluator analyzing the "Drift" of an agent.
The agent was given a global game objective (Game Goal).
At a specific time step, the agent was probed for its current internal intentions (Agent's Stated Intentions).

Your task is to determine whether the agent's current intentions are still aligned with the global game objective, or if the agent has completely drifted (hallucinated a new goal, gotten permanently distracted by an irrelevant sub-task, or forgotten the original objective).

Respond with ONLY the number 0 or 1:
0 = NO DRIFT (The agent's intentions are aligned with and actively contributing to the Game Goal)
1 = DRIFT DETECTED (The agent's intentions are disconnected, contradictory, or completely irrelevant to the Game Goal)
"""

async def evaluate_drift(model, game_goal: str, agent_intentions: str) -> int:
    """Asks the LLM Judge if the agent has drifted."""
    if not agent_intentions or not game_goal:
        return 0 # Default to 0 if data is missing
        
    prompt = f"Game Goal:\n{game_goal}\n\nAgent's Stated Intentions:\n{agent_intentions}"
    
    try:
        response = await model.generate(prompt, system=JUDGE_SYSTEM_PROMPT)
        score_text = response.completion.strip()
        
        if "1" in score_text:
            return 1
        return 0
    except Exception as e:
        print(f"Error calling LLM judge: {e}")
        return 0

async def main(run_name: str, model_name: str = "google/gemini-2.5-flash"):
    results_dir = f"experiments/results/2026-06-26-error-category-game-score-experiment/{run_name}"
    csv_path = os.path.join(results_dir, "parsed_trajectory_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run parse_ablation_results.py first.")
        return
        
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if "Drift_Completion" not in df.columns or "Drift_Game_Goal" not in df.columns:
        print("Error: DataFrame is missing Drift columns. Ensure parse_ablation_results.py is up to date.")
        return
        
    print(f"Initializing LLM Judge ({model_name})...")
    # Load API key if available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and os.path.exists("secrets/gemini-key"):
        with open("secrets/gemini-key", "r") as f:
            api_key = f.read().strip()
            os.environ["GEMINI_API_KEY"] = api_key
            
    model = get_model(model_name)
    
    # We will evaluate drift for rows where Drift_Completion exists and is not nan
    # Since evaluating every single step of every seed could be thousands of API calls,
    # we should do this concurrently.
    
    df["Drift_Score"] = 0
    
    tasks = []
    indices = []
    
    for idx, row in df.iterrows():
        drift_completion = str(row["Drift_Completion"])
        drift_game_goal = str(row["Drift_Game_Goal"])
        
        if pd.isna(row["Drift_Completion"]) or drift_completion.strip() == "":
            continue
            
        indices.append(idx)
        tasks.append(evaluate_drift(model, drift_game_goal, drift_completion))
        
    print(f"Evaluating {len(tasks)} steps for Control-Path Drift...")
    
    # Run in batches of 50 to avoid rate limits
    batch_size = 50
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        print(f"Processed {len(results)}/{len(tasks)}...")
        
    for idx, res in zip(indices, results):
        df.at[idx, "Drift_Score"] = res
        
    # Recalculate Cumulative Drift
    df["Cumulative_Drift"] = df.groupby(["Config", "Seed"])["Drift_Score"].cumsum()
    
    out_csv = os.path.join(results_dir, "parsed_trajectory_data_with_drift.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved drift analysis to {out_csv}")
    
    # Plotting Drift
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Step", y="Cumulative_Drift", hue="Config", errorbar="ci")
        plt.title("Cumulative Control-Path Drift over Horizon")
        plt.xlabel("Step Index (Proxy for Horizon)")
        plt.ylabel("Cumulative Drift Occurrences")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "cumulative_drift_vs_step.png"), dpi=300)
        plt.close()
        print("Plotted Cumulative Drift.")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="The exact run name (including timestamp) to analyze")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash", help="The LLM judge to use")
    args = parser.parse_args()
    
    asyncio.run(main(args.run_name, args.model))
