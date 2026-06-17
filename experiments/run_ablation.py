#!/usr/bin/env python3
import argparse
import sys

from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample

sys.path.append(".")

from environments.cooking import CookingWorldEnvironment
from harness.mechanisms import (
    M1InfoSeeking,
    M2Memory,
    M3StateExternalization,
    M4AdaptiveCompute,
    M5ActionTemplating,
    M6Planning,
)
from harness.solver import harness_orchestrator
from harness.scorer import harness_scorer


def create_tasks(experiment_name: str, active_mechanisms: list, seeds: int, steps: int):
    if not active_mechanisms:
        config_name = "baseline"
    else:
        config_name = "_".join([m.__class__.__name__[:2] for m in active_mechanisms]).lower()

    tasks = []
    for idx in range(seeds):
        current_seed = 1000 + idx
        
        @task(name=f"{experiment_name}_{config_name}_seed_{current_seed}")
        def ablation_task() -> Task:
            dataset = MemoryDataset([
                Sample(
                    input="Initialize CookingWorld Run with unique layout configurations.", 
                    target="Success", 
                    metadata={"seed": current_seed}
                )
            ])
            return Task(
                dataset=dataset,
                solver=harness_orchestrator(
                    environment_factory=CookingWorldEnvironment, 
                    mechanisms=active_mechanisms, 
                    max_steps=steps,
                    seed=current_seed
                ),
                scorer=harness_scorer()
            )
        tasks.append(ablation_task())
    return tasks

def main() -> None:
    parser = argparse.ArgumentParser(description="Complete Ablation CLI for Long-Horizon Game Agents.")
    
    parser.add_argument("--experiment", type=str, default="run", help="Prefix name for the experiment to group tasks in Inspect View")
    
    parser.add_argument("--m1", action="store_true", help="Enable M1 Information-Seeking mode")
    
    parser.add_argument("--m2", action="store_true", help="Enable M2 Context/Memory Mechanism")
    parser.add_argument("--m3", action="store_true", help="Enable M3 State Externalization Mechanism")
    parser.add_argument("--m4", action="store_true", help="Enable M4 Adaptive Compute Parameter Adjustments")
    parser.add_argument("--m5", action="store_true", help="Enable M5 Action Templating Constraints")
    parser.add_argument("--m6", action="store_true", help="Enable M6 Planning & Synthesis (Map-then-Act)")
    
    parser.add_argument("--model", type=str, default="openai/gpt-4o", help="Target model provider path")
    parser.add_argument("--steps", type=int, default=30, help="Maximum execution steps per run")
    parser.add_argument("--seeds", type=int, default=5, help="Number of unique game seeds to evaluate")
    
    args = parser.parse_args()

    active_mechanisms = []
    
    if args.m1:
        active_mechanisms.append(M1InfoSeeking())
    if args.m2:
        active_mechanisms.append(M2Memory())
    if args.m3:
        active_mechanisms.append(M3StateExternalization())
    if args.m4:
        active_mechanisms.append(M4AdaptiveCompute())
    if args.m5:
        active_mechanisms.append(M5ActionTemplating())
    if args.m6:
        active_mechanisms.append(M6Planning())

    tasks = create_tasks(args.experiment, active_mechanisms, args.seeds, args.steps)

    mechanism_names = [m.name for m in active_mechanisms]
    print("=" * 60)
    print("LAUNCHING ABLATION MATRIX")
    print(f"Model Target:      {args.model}")
    print(f"Max Run Steps:     {args.steps}")
    print(f"Evaluation Seeds:  {args.seeds}")
    print(f"Active Harnesses:  {mechanism_names if mechanism_names else 'No harnesses'}")
    print("=" * 60)
    
    eval(tasks, model=args.model)

if __name__ == "__main__":
    main()
