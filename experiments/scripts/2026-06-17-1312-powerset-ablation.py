#!/usr/bin/env python3
import sys
import itertools
from inspect_ai import eval
import json

sys.path.append(".")

import os
try:
    with open("secrets/openrouter-key", "r") as f:
        os.environ["OPENROUTER_API_KEY"] = f.read().strip()
except FileNotFoundError:
    pass

from harness.mechanisms import (
    M2Memory,
    M3StateExternalization,
    M5ActionTemplating,
    M6Planning,
)
from experiments.run_ablation import create_tasks

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

# TODO: How do levels work?
def main():
    model = "openrouter/meta-llama/llama-3.2-3b-instruct:free"
    seeds = 10
    steps = 200
    experiment_name = "2026-06-17-1312-powerset"
    
    # always give valid action support
    base_mechanisms = [M5ActionTemplating]
    
    variable_mechanisms = [M2Memory, M3StateExternalization, M6Planning]
    
    all_tasks = []
    
    for combo in powerset(variable_mechanisms):
        active_mechanisms = [m() for m in base_mechanisms] + [m() for m in combo]
        
        tasks = create_tasks(experiment_name, active_mechanisms, seeds, steps)
        all_tasks.extend(tasks)
            
    print("=" * 60)
    print("LAUNCHING MASSIVELY PARALLEL POWERSET MATRIX")
    print(f"Model Target:      {model}")
    print(f"Max Run Steps:     {steps}")
    print(f"Evaluation Seeds:  {seeds} (Samples per Config)")
    print(f"Total Tasks:       {len(all_tasks)}")
    print("=" * 60)
    
    eval_logs = eval(all_tasks, model=model, max_tasks=2)
    
    
    dataset_records = []
    for log in eval_logs:
        task_name = log.eval.task
        
        for sample in log.samples:
            meta = sample.metadata if sample.metadata else {}
            
            telemetry = meta.get("trajectory_telemetry", [])
            # TODO: Probably don't want to default to 0? Why would final_score not be set
            final_score = meta.get("final_score", 0.0)
            seed = meta.get("seed", None)
            
            dataset_records.append({
                "task": task_name,
                "seed": seed,
                "score": final_score,
                # TODO: Is this an accurate way to find the number of steps?
                "steps_taken": len(telemetry),
                # Do I want to store this? What's in telemetry?
                "trajectory": telemetry
            })
            
    dataset_path = "experiments/powerset_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset_records, f, indent=2)
        
    print("=" * 60)
    print(f"POWERSET COMPLETE! Consolidated dataset saved to:")
    print(f" -> {dataset_path}")
    print(f"Total Records Extracted: {len(dataset_records)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
