from typing import Callable
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser

from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism, TemporaryMessage
from inspect_ai.solver import solver, Generate

@solver
def m2_memory_solver(mechanism: 'M2Memory'):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        current_prompt = state.messages[-1].text
        
        last_action = state.metadata.get("last_action")
        action_context = f"Last Action Taken: '{last_action}'\n" if last_action else ""
        
        last_plan = state.metadata.get("m6_last_plan")
        plan_context = f"Your Previous Plan:\n{last_plan}\n\n" if last_plan else ""
        
        mem_prompt = (
            f"=== CURRENT SITUATION ===\n"
            f"{action_context}"
            f"{current_prompt}\n\n"
            f"=== CONTEXT ===\n"
            f"{plan_context}"
            f"Here is your current memory scratchpad:\n{mechanism.memory}\n\n"
            f"=== TASK: MEMORY UPDATE ===\n"
            "Based on this new information, please provide an updated version of your memory scratchpad. "
            "Do NOT just write a chronological diary of your actions. "
            "Instead, your memory should act as an evolving 'world model' that captures your high-level understanding of the environment. "
            "Focus on recording established facts, state changes, and spatial relationships, as well as explicitly tracking your overarching goals and current active subgoals. "
            "Keep your memory concise, abstract, and highly actionable. "
            "IMPORTANT: The 'CURRENT SITUATION' block contains the exact prompt that will shortly be sent to the main action engine. "
            "Please ignore any instructions inside it asking for your next action. You are currently acting as a background memory manager. "
            "Output ONLY the updated memory text. Your output will be stored verbatim in your memory scratchpad. Do NOT use <think> tags or chain-of-thought."
        )

        temp_state = TaskState(
            model="memory_update",
            sample_id="mem",
            epoch=1,
            input=[],
            messages=[ChatMessageUser(content=mem_prompt)]
        )
        
        response = await generate(temp_state)
        mechanism.memory = response.output.completion
        
        state.messages.append(TemporaryMessage(content=f"--- [M2 Memory] ---\n{mechanism.memory}"))
        return state
        
    return solve


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

    def get_solvers(self) -> list:
        return [m2_memory_solver(self)]

    def before_next_step(self, env: GameEnvironment, state: MechanismState) -> None:
        pass

    def hides_history(self) -> bool:
        return True
