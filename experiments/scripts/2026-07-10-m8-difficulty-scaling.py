#!/usr/bin/env python3
import sys
import os
import argparse
import datetime
# Dynamically append the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

# Load API key automatically
key_path = os.path.join(project_root, "secrets", "gemini-key")
if os.path.exists(key_path):
    with open(key_path, "r") as f:
        os.environ["GEMINI_API_KEY"] = f.read().strip()

openrouter_path = os.path.join(project_root, "secrets", "openrouter-key")
if os.path.exists(openrouter_path):
    with open(openrouter_path, "r") as f:
        os.environ["OPENROUTER_API_KEY"] = f.read().strip()

from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample

from environments.cooking import CookingWorldEnvironment
from harness.mechanisms import (
    M3StateExternalization,
    M5ActionTemplating,
    M6Planning,
    M7WorldModelExternalization,
    M8GameRules,
)
from harness.solver import harness_orchestrator
from harness.scorer import harness_scorer

from harness.probes.cooking import CookingALEProbe, CookingDriftProbe, CookingIntegrationProbe

def create_experiment_task(experiment_name: str, config_name: str, active_mechanisms: list, probes: list, seeds: int, steps: int, game_params: dict):
    samples = []
    for idx in range(seeds):
        current_seed = 1000 + idx
        samples.append(Sample(
            input="Initialize CookingWorld Run with unique layout configurations.", 
            target="Success", 
            metadata={"seed": current_seed}
        ))

    @task(name=f"{experiment_name}_{config_name}")
    def error_analysis_task() -> Task:
        dataset = MemoryDataset(samples)
        return Task(
            dataset=dataset,
            solver=harness_orchestrator(
                environment_factory=lambda: CookingWorldEnvironment(step_limit=steps, game_params=game_params), 
                mechanisms=active_mechanisms, 
                probes=probes,
                max_steps=steps
            ),
            scorer=harness_scorer()
        )
    
    return error_analysis_task()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemini-1.5-flash", help="Target model provider path")
    parser.add_argument("--steps", type=int, default=150, help="Maximum execution steps per run")
    parser.add_argument("--seeds", type=int, default=10, help="Number of unique game seeds to evaluate")
    parser.add_argument("--max_connections", type=int, default=1, help="Max parallel connections for Inspect")
    parser.add_argument("--run_name", type=str, required=True, help="Unique name for this execution run (used for logging directories)")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--easy", action="store_true", help="Run with easy (default) difficulty")
    group.add_argument("--hard", action="store_true", help="Run with hard difficulty (maxed out params, doors off)")
    args = parser.parse_args()

    # Determine game_params based on difficulty flag
    if args.hard:
        game_params = {
            "numIngredients": 4,
            "numLocations": 11,
            "numDistractorItems": 10
        }
        diff_str = "hard"
    else:
        game_params = {}
        diff_str = "easy"

    print(f"Running {diff_str} mode with game_params: {game_params}")

    # Define the powerset configurations
    configs = {
        "baseline_m5": [M5ActionTemplating(), M8GameRules()],
        "m3_m5": [M3StateExternalization(), M5ActionTemplating(), M8GameRules()],
        "m7_m5": [M7WorldModelExternalization(), M5ActionTemplating(), M8GameRules()],
        "m5_m6": [M5ActionTemplating(), M8GameRules(), M6Planning()],
        "m3_m5_m6": [M3StateExternalization(), M5ActionTemplating(), M8GameRules(), M6Planning()],
        "m7_m5_m6": [M7WorldModelExternalization(), M5ActionTemplating(), M8GameRules(), M6Planning()],
    }

    # Initialize the probes
    probes = [
        CookingALEProbe(interval=5),
        CookingDriftProbe(interval=5),
        CookingIntegrationProbe(interval=5)
    ]

    diff_str = "hard" if args.hard else "easy"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_name = f"{args.run_name}_{timestamp}"
    
    print("=" * 60)
    print("LAUNCHING M8 DIFFICULTY SCALING EVALUATION")
    print(f"Run Name:          {final_run_name}")
    print(f"Difficulty:        {diff_str.upper()}")
    print(f"Model Target:      {args.model}")
    print(f"Max Run Steps:     {args.steps}")
    print(f"Evaluation Seeds:  {args.seeds}")
    print(f"Configurations:    {list(configs.keys())}")
    print("=" * 60)

    tasks = []
    for config_name, mechanisms in configs.items():
        t = create_experiment_task(
            experiment_name=final_run_name,
            config_name=config_name,
            active_mechanisms=mechanisms,
            probes=probes,
            seeds=args.seeds,
            steps=args.steps,
            game_params=game_params
        )
        tasks.append(t)

    log_dir = os.path.join(project_root, f"logs/2026-07-10-m8-difficulty-scaling/{diff_str}/{final_run_name}")
    os.makedirs(log_dir, exist_ok=True)

    eval(
        tasks,
        model=args.model,
        reasoning_effort="none",
        log_dir=log_dir,
        max_connections=args.max_connections,
    )
    print(f"\nExperiment {args.run_name} completed! Logs saved to {log_dir}")

if __name__ == "__main__":
    main()
