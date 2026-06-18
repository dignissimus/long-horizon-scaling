from typing import Callable
from inspect_ai.solver import TaskState, solver, Generate
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism, TemporaryMessage

@solver
def m6_planning_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        current_prompt = state.messages[-1].text
        last_plan = state.metadata.get("m6_last_plan", "No previous plan.")
        
        planning_instruction = (
            f"Here is your previous plan:\n{last_plan}\n\n"
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

        response = await generate(temp_state)
        plan_output = response.output.completion

        state.metadata["m6_last_plan"] = plan_output
        state.messages.append(TemporaryMessage(content=f"--- [M6 Planning & Synthesis] ---\n{plan_output}"))
        return state
    return solve

@solver
def m6_planning_cot_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        current_prompt = state.messages[-1].text
        # TODO: Don't store in metadata
        last_plan = state.metadata.get("m6_last_plan", "No previous plan.")
        
        cot_instruction = (
            f"Here is your previous plan:\n{last_plan}\n\n"
            f"Here is the current situation:\n{current_prompt}\n\n"
            "Before devising a formal plan, please think step-by-step about what you know, what you have, and what your ultimate goal should be. Output ONLY your reasoning."
        )

        temp_state = TaskState(
            model="planning",
            sample_id="plan",
            epoch=1,
            input=[],
            messages=[ChatMessageUser(content=cot_instruction)]
        )

        response1 = await generate(temp_state)
        reasoning_output = response1.output.completion

        formatting_instruction = (
            f"Here is your reasoning:\n{reasoning_output}\n\n"
            "Now, synthesize your reasoning into a formal plan. Output your response exactly in this format:\n"
            "KNOWN: [what you know about the environment]\n"
            "HAVE: [what is in your inventory]\n"
            "GOAL: [what you are trying to do]\n"
            "PLAN: [your step-by-step plan for the next 3 actions]"
        )

        temp_state2 = TaskState(
            model="planning",
            sample_id="plan",
            epoch=1,
            input=[],
            messages=[ChatMessageUser(content=formatting_instruction)]
        )

        response2 = await generate(temp_state2)
        plan_output = response2.output.completion

        state.metadata["m6_last_plan"] = plan_output
        state.messages.append(TemporaryMessage(content=f"--- [M6 Planning & Synthesis] ---\nReasoning:\n{reasoning_output}\n\n{plan_output}"))
        return state
    return solve

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

# TODO: How to decide if impl in get_solvers or augment
class M6PlanningForced(Mechanism):
    """
    Targets: integration.
    Forces an explicit synthesis step before action execution (Map-then-Act)
    using a secondary asynchronous LLM call.
    """
    def get_solvers(self) -> list:
        return [m6_planning_solver()]

class M6PlanningForcedWithChainOfThought(Mechanism):
    """
    Targets: integration.
    Forces an explicit synthesis step before action execution (Map-then-Act)
    using a secondary asynchronous LLM call.
    """
    def get_solvers(self) -> list:
        return [m6_planning_cot_solver()]

M6Planning = M6PlanningForcedWithChainOfThought
