#!/usr/bin/env python3
import sys
import os
import argparse

# Dynamically append the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

# Load API key automatically
key_path = os.path.join(project_root, "secrets", "gemini-key")
if os.path.exists(key_path):
    with open(key_path, "r") as f:
        os.environ["GEMINI_API_KEY"] = f.read().strip()

from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample

from environments.cooking import CookingWorldEnvironment
from harness.mechanisms import (
    M3StateExternalization,
    M5ActionTemplating,
    M6Planning,
    M7WorldModelExternalization,
)
from harness.solver import harness_orchestrator
from harness.scorer import harness_scorer

from harness.probes.cooking import CookingALEProbe, CookingDriftProbe, CookingIntegrationProbe

def create_experiment_task(experiment_name: str, config_name: str, active_mechanisms: list, probes: list, seeds: int, steps: int):
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
                environment_factory=lambda: CookingWorldEnvironment(step_limit=steps), 
                mechanisms=active_mechanisms, 
                probes=probes,
                max_steps=steps
            ),
            scorer=harness_scorer()
        )
    
    return error_analysis_task()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash", help="Target model provider path")
    parser.add_argument("--steps", type=int, default=100, help="Maximum execution steps per run")
    parser.add_argument("--seeds", type=int, default=10, help="Number of unique game seeds to evaluate")
    parser.add_argument("--max_connections", type=int, default=1, help="Max parallel connections for Inspect")
    parser.add_argument("--run_name", type=str, required=True, help="Unique name for this execution run (used for logging directories)")
    args = parser.parse_args()

    # Define the powerset configurations
    configs = {
        "baseline_m5": [M5ActionTemplating()],
        "m3_m5": [M3StateExternalization(), M5ActionTemplating()],
        "m7_m5": [M7WorldModelExternalization(), M5ActionTemplating()],
        "m5_m6": [M5ActionTemplating(), M6Planning()],
        "m3_m5_m6": [M3StateExternalization(), M5ActionTemplating(), M6Planning()],
        "m7_m5_m6": [M7WorldModelExternalization(), M5ActionTemplating(), M6Planning()],
    }

    # Initialize the probes
    probes = [
        CookingALEProbe(interval=5),
        CookingDriftProbe(interval=5),
        CookingIntegrationProbe(interval=5)
    ]

    tasks = []
    for config_name, mechanisms in configs.items():
        t = create_experiment_task(
            experiment_name=args.run_name,
            config_name=config_name,
            active_mechanisms=mechanisms,
            probes=probes,
            seeds=args.seeds,
            steps=args.steps
        )
        tasks.append(t)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_name = f"{args.run_name}_{timestamp}"
    
    print("=" * 60)
    print("LAUNCHING LONG HORIZON ERROR ANALYSIS")
    print(f"Run Name:          {final_run_name}")
    print(f"Model Target:      {args.model}")
    print(f"Max Run Steps:     {args.steps}")
    print(f"Evaluation Seeds:  {args.seeds}")
    print(f"Configurations:    {list(configs.keys())}")
    print(f"Probes Active:     {[p.name for p in probes]}")
    print("=" * 60)
    
    # Force max_connections=1 to prevent RPM rate limit blowouts on Gemini Flash 
    # due to massive API call expansion from interval=1 probes
    log_dir = f"logs/2026-06-26-error-category-game-score-experiment/{final_run_name}"
    eval(tasks, model=args.model, max_connections=args.max_connections, log_dir=log_dir)

if __name__ == "__main__":
    main()
