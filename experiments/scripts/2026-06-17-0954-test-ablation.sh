#!/bin/bash


export OPENROUTER_API_KEY=$(cat secrets/openrouter-key)

MODEL="openrouter/meta-llama/llama-3.2-3b-instruct"

echo "Checking Java version inside script:"
java -version
which java

echo "Running Baseline (without Memory Harness, but with M5 Action Listing)..."
uv run python experiments/run_ablation.py --experiment test --model "$MODEL" --m5 --steps 3 --seeds 1

echo ""
echo "Running with Memory Harness (--m2) and Action Listing (--m5)..."
uv run python experiments/run_ablation.py --experiment test --model "$MODEL" --m2 --m5 --steps 3 --seeds 1

echo ""
echo "Running with Planning (--m6) and Action Listing (--m5)..."
uv run python experiments/run_ablation.py --experiment test --model "$MODEL" --m6 --m5 --steps 3 --seeds 1
