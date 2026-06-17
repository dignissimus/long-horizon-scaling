from inspect_ai.scorer import scorer, Score, Target, mean
from inspect_ai.solver import TaskState

@scorer(metrics=[mean()])
def harness_scorer():
    """Scores the agent based on the final environment score exported by the harness."""
    async def score(state: TaskState, target: Target) -> Score:
        final_score = state.metadata.get("final_score", 0.0)
        return Score(
            value=final_score,
            answer=state.output.completion if state.output else "Done"
        )
    return score
