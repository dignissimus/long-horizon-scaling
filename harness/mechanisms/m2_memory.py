from typing import Callable
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser

from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism


class M2Memory(Mechanism):
    """
    Targets: drift.
    Maintains a rolling context of what has been seen and tried to prevent loops and goal drift.
    Updates its memory using the LLM before generating the action.
    """
    def __init__(self) -> None:
        super().__init__()
        self.memory = "I have just started. I have no memories yet."

    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        return current_prompt

    async def augment_prompt_async(self, current_prompt: str, env: GameEnvironment, state: MechanismState, generate_fn: Callable) -> str:
        mem_prompt = (
            f"Here is the current situation:\n{current_prompt}\n\n"
            f"Here is your current memory scratchpad:\n{self.memory}\n\n"
            "Based on this new information, please provide an updated version of your memory scratchpad. "
            "It should be a concise summary of what you have done so far, what your current sub-goal is, and any important details you need to remember. "
            "Output ONLY the updated memory text."
        )
        
        temp_state = TaskState(
            model="memory_update",
            sample_id="mem",
            epoch=1,
            input=[],
            messages=[ChatMessageUser(content=mem_prompt)]
        )
        
        response = await generate_fn(temp_state)
        self.memory = response.output.completion
        
        return f"{current_prompt}\n\n--- [M2 Memory] ---\n{self.memory}\n"

    def before_next_step(self, env: GameEnvironment, state: MechanismState) -> None:
        pass
