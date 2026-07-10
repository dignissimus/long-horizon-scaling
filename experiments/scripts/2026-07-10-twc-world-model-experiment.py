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

from environments.twc import TextWorldCommonSenseEnvironment
from environments.simonsays import SimonSaysEnvironment
from environments.sorting import SortingEnvironment
from environments.arithmetic import ArithmeticEnvironment
from environments.mapreader import MapReaderEnvironment

from harness.mechanisms import (
    M2Memory,
    M3StateExternalization,
    M5ActionTemplating,
    M6Planning,
    M7WorldModelExternalization,
    M8GameRules,
)
from harness.solver import harness_orchestrator
from harness.scorer import harness_scorer
from harness.probes.generic import GenericDriftProbe, GenericIntegrationProbe

def create_experiment_task(experiment_name: str, config_name: str, env_name: str, env_class, active_mechanisms: list, seeds: int, steps: int, game_params: dict):
    samples = []
    for idx in range(seeds):
        current_seed = 1000 + idx
        samples.append(Sample(
            input=f"Initialize {env_name} Run.", 
            target="Success", 
            metadata={"seed": current_seed}
        ))

    @task(name=f"{experiment_name}_{env_name}_{config_name}")
    def generic_game_task() -> Task:
        dataset = MemoryDataset(samples)
        return Task(
            dataset=dataset,
            solver=harness_orchestrator(
                environment_factory=lambda: env_class(step_limit=steps, game_params=game_params), 
                mechanisms=active_mechanisms, 
                probes=[
                    GenericDriftProbe(interval=5),
                    GenericIntegrationProbe(interval=5)
                ],
                max_steps=steps
            ),
            scorer=harness_scorer()
        )
    
    return generic_game_task()

def main():
    parser = argparse.ArgumentParser()
    # Using the standard ultra-cheap model
    parser.add_argument("--model", type=str, default="openrouter/openai/gpt-4o-mini", help="Target model provider path")
    parser.add_argument("--steps", type=int, default=50, help="Maximum execution steps per run")
    parser.add_argument("--seeds", type=int, default=5, help="Number of unique game seeds to evaluate")
    parser.add_argument("--max_connections", type=int, default=1, help="Max parallel connections for Inspect")
    parser.add_argument("--run_name", type=str, required=True, help="Unique name for this execution run")
    parser.add_argument("--hard", action="store_true", help="Enable TWC hard mode (doors and inventory limit)")
    args = parser.parse_args()

    # Configure difficulty parameters for TWC
    game_params = {}
    diff_str = "easy"
    if args.hard:
        game_params["includeDoors"] = 0
        game_params["limitInventorySize"] = 1
        diff_str = "hard"
    else:
        game_params["includeDoors"] = 0
        game_params["limitInventorySize"] = 0

    # Define the powerset configurations, making M3 and M7 mutually exclusive
    configs = {
        "baseline_m5": [M5ActionTemplating(), M8GameRules()],
        "m3_m5": [M3StateExternalization(), M5ActionTemplating(), M8GameRules()],
        "m7_m5": [M7WorldModelExternalization(), M5ActionTemplating(), M8GameRules()],
        "m5_m6": [M5ActionTemplating(), M8GameRules(), M6Planning()],
        "m3_m5_m6": [M3StateExternalization(), M5ActionTemplating(), M8GameRules(), M6Planning()],
        "m7_m5_m6": [M7WorldModelExternalization(), M5ActionTemplating(), M8GameRules(), M6Planning()],
        "m2_m5": [M2Memory(), M5ActionTemplating(), M8GameRules()],
        "m2_m3_m5": [M2Memory(), M3StateExternalization(), M5ActionTemplating(), M8GameRules()],
        "m2_m7_m5": [M2Memory(), M7WorldModelExternalization(), M5ActionTemplating(), M8GameRules()],
        "m2_m5_m6": [M2Memory(), M5ActionTemplating(), M8GameRules(), M6Planning()],
        "m2_m3_m5_m6": [M2Memory(), M3StateExternalization(), M5ActionTemplating(), M8GameRules(), M6Planning()],
        "m2_m7_m5_m6": [M2Memory(), M7WorldModelExternalization(), M5ActionTemplating(), M8GameRules(), M6Planning()],
    }

    environments = {
        "twc": TextWorldCommonSenseEnvironment
    }

    tasks = []
    for env_name, env_class in environments.items():
        for config_name, mechanisms in configs.items():
            t = create_experiment_task(
                experiment_name=args.run_name,
                config_name=config_name,
                env_name=env_name,
                env_class=env_class,
                active_mechanisms=mechanisms,
                seeds=args.seeds,
                steps=args.steps,
                game_params=game_params
            )
            tasks.append(t)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_name = f"{args.run_name}_{timestamp}"
    
    print("=" * 60)
    print("LAUNCHING MULTI-ENVIRONMENT SCALING EVALUATION")
    print(f"Run Name:          {final_run_name}")
    print(f"Model Target:      {args.model}")
    print(f"Max Run Steps:     {args.steps}")
    print(f"Evaluation Seeds:  {args.seeds}")
    print(f"Configurations:    {list(configs.keys())}")
    print(f"Difficulty Mode:   {diff_str.upper()}")
    print("=" * 60)
    
    log_dir = os.path.join(project_root, f"logs/2026-07-10-twc-world-model-experiment/{diff_str}/{final_run_name}")
    os.makedirs(log_dir, exist_ok=True)
    eval(tasks, model=args.model, reasoning_effort=None, max_connections=args.max_connections, log_dir=log_dir)

if __name__ == "__main__":
    main()
