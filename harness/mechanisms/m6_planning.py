from typing import Callable
from inspect_ai.solver import TaskState
from inspect_ai.model import ChatMessageUser

from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism

class M6PlanningPrompt(Mechanism):
    """
    Targets: integration.
    Enforces the Map-Then-Act pattern via a strict CoT template output expectation in the prompt.
    """
    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        # TODO: This has problems as this instructs the agent to output in a format that's invalid
        raise NotImplementedError
        planning_instruction = (
            "\n--- [M6 Planning & Synthesis] ---\n"
            "Before writing your final command, synthesize your current situation by outputting exactly:\n"
            "KNOWN: [what you know]\n"
            "HAVE: [your inventory]\n"
            "GOAL: [what you are trying to do]\n"
            "PLAN: [next 3 steps]\n"
            "ACTION: [your final command to execute]"
        )
        return current_prompt + planning_instruction

class M6PlanningForced(Mechanism):
    """
    Targets: integration.
    Forces an explicit synthesis step before action execution (Map-then-Act)
    using a secondary asynchronous LLM call.
    """
    async def augment_prompt_async(self, current_prompt: str, env: GameEnvironment, state: MechanismState, generate_fn: Callable) -> str:
        planning_instruction = (
            f"Here is the current situation:\n{current_prompt}\n\n"
            "Please synthesize the situation and create a plan. Output your response exactly in this format:\n"
            "KNOWN: [what you know about the environment]\n"
            "HAVE: [what is in your inventory]\n"
            "GOAL: [what you are trying to do]\n"
            "PLAN: [your step-by-step plan for the next 3 actions]"
        )
        
        temp_state = TaskState(
            model="planning",
            sample_id="plan",
            epoch=1,
            input=[],
            messages=[ChatMessageUser(content=planning_instruction)]
        )
        
        response = await generate_fn(temp_state)
        plan_output = response.output.completion
        
        return f"{current_prompt}\n\n--- [M6 Planning & Synthesis] ---\n{plan_output}\n"

M6Planning = M6PlanningForced
