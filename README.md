# Long-Horizon Scaling

## Usage Example

You can run a simple ablation experiment using the `run_ablation.py` script. Here is a simplified example of how to run an experiment with memory and action templating enabled

```bash
uv run python experiments/run_ablation.py \
  --experiment my_experiment \
  --model openai/gpt-4o \
  --m2 \
  --m5 \
  --steps 100 \
  --seeds 3
```

### Available Mechanisms:
- `--m2`: Memory
- `--m3`: State Externalization
- `--m5`: Action Templating
- `--m6`: Planning & Synthesis

For a complete list of arguments, run `uv run python experiments/run_ablation.py --help`.

## Python Example

You can also run experiments programmatically by importing the necessary components:

```python
from inspect_ai import eval
from harness.mechanisms import M2Memory, M5ActionTemplating
from experiments.run_ablation import create_tasks

# Define the mechanisms to test
active_mechanisms = [M2Memory(), M5ActionTemplating()]

# Create evaluation tasks
tasks = create_tasks(
    experiment_name="python_example",
    active_mechanisms=active_mechanisms,
    seeds=3,
    steps=100
)

# Run the evaluation
eval(tasks, model="openai/gpt-4o")
```
