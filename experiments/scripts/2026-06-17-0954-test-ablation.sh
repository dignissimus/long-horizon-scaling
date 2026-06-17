#!/bin/bash


export OPENROUTER_API_KEY=$(cat secrets/openrouter-key)

MODEL="openrouter/meta-llama/llama-3.2-3b-instruct"

echo "Checking Java version inside script:"
java -version
which java

echo "Running Baseline (without Memory Harness)..."
uv run python experiments/run_ablation.py --model "$MODEL" --steps 1 --seeds 1

echo ""
echo "Running with Memory Harness (--m2)..."
uv run python experiments/run_ablation.py --model "$MODEL" --m2
