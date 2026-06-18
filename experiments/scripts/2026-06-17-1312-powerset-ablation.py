#!/usr/bin/env python3
import os
import sys
import json
import itertools
import argparse
from datetime import datetime
from inspect_ai import eval

sys.path.append(".")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--max-connections", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=1)
    args = parser.parse_args()

    key_path = os.path.join("secrets", "gemini-key")
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            os.environ["GEMINI_API_KEY"] = f.read().strip()

    model = args.model
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
    
    eval_logs = eval(all_tasks, model=model, max_connections=args.max_connections, max_tasks=args.max_tasks, reasoning_effort="none", display="plain")
    
    
    dataset_records = []
    for log in eval_logs:
        task_name = log.eval.task
        
        for sample in log.samples:
            meta = sample.metadata if sample.metadata else {}
            
            telemetry = meta.get("trajectory_telemetry", [])
            
            dataset_records.append({
                "task": log.eval.task,
                "mechanism_config": log.eval.task.replace("2026-06-17-1312-powerset_", ""),
                "model": model,
                "seed": sample.metadata.get("seed", None),
                # TODO: Not sure we should default to 0
                "score": meta.get("final_score", 0.0),
                # TODO: Is this an accurate way to find the number of steps?
                "steps_taken": len(telemetry),
                "history": [msg.model_dump() for msg in sample.messages],
                # Do I want to store this? What's in telemetry?
                "trajectory": telemetry
            })
            
    os.makedirs("experiments/experiment-output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_slug = model.split("/")[-1].replace(":", "-")
    output_path = f"experiments/experiment-output/powerset_dataset_{model_slug}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(dataset_records, f, indent=2)

    print("=" * 60)
    print("POWERSET COMPLETE! Consolidated dataset saved to:")
    print(f" -> {output_path}")
    print(f"Total Records Extracted: {len(dataset_records)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
