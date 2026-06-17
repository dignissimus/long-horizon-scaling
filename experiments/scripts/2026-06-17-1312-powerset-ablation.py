#!/usr/bin/env python3
import sys
import itertools
from inspect_ai import eval

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

def main():
    model = "openrouter/meta-llama/llama-3.2-3b-instruct"
    seeds = 10
    steps = 1000
    experiment_name = "2026-06-17-1312-powerset-ablation"
    
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
    print(f"Evaluation Seeds:  {seeds}")
    print(f"Total Tasks:       {len(all_tasks)} (8 Configs x {seeds} Seeds)")
    print("=" * 60)
    
    # Pass all 80 tasks into a single eval call to run them perfectly in parallel!
    eval(all_tasks, model=model)

if __name__ == "__main__":
    main()
