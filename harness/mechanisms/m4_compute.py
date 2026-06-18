from inspect_ai.solver import Solver, TaskState, Generate
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

from harness.mechanisms.mechanism import Mechanism


def explicit_cot_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.append(ChatMessageUser(content="Before taking your action, please think step-by-step about what you should do next. Output ONLY your reasoning process and do not include the action itself. You may use <think> tags in your reasoning."))
        response = await generate(state)
        state.messages.append(ChatMessageAssistant(content=response.output.completion))
        state.messages.append(ChatMessageUser(content="Based on your reasoning, what is your next action? Do NOT wrap the action in quotes. IMPORTANT: Your output will be executed verbatim by the game engine. Do NOT use <think> tags or chain-of-thought."))
        return state
    return solve

# TODO: Moved COT to M6 since the spirit of M4 is dynamically controlling how much effort we spend
class M4ChainOfThought(Mechanism):
    """
    Targets: budget/liveness.
    Enables explicit 2-step multi-turn chain_of_thought
    """
    def get_solvers(self) -> list[Solver]:
        raise NotImplementedError
        return [explicit_cot_solver()]

M4AdaptiveCompute = M4ChainOfThought
