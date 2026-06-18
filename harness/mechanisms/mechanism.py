from typing import Any, Callable
from inspect_ai.solver import Solver

from harness.actions.types import ActionResult, MechanismState, TakeAction
from harness.environment.environment import GameEnvironment
from inspect_ai.model import ChatMessageUser

class TemporaryMessage(ChatMessageUser):
    """A user message that is stripped from the history at the end of the step."""
    pass


class Mechanism:
    """
    Base Class for all Harness Mechanisms.
    Implements a passive pass-through for all lifecycle hooks.
    """
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def configure_model(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Modify inference or generation parameters (e.g., temperature, system instructions)."""
        return parameters

    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        """Appends static/deterministic information to the user prompt string."""
        return current_prompt

    async def augment_prompt_async(self, current_prompt: str, env: GameEnvironment, state: MechanismState, generate_fn: Callable) -> str:
        """Asynchronously executes secondary LLM calls to build plans, thoughts, or reflections."""
        return current_prompt

    def after_generation(self, model_response: str, env: GameEnvironment, state: MechanismState) -> ActionResult:
        """Inspects the generated model output and chooses to pass it through or force a retry."""
        return TakeAction(action_string=model_response)

    def before_next_step(self, env: GameEnvironment, state: MechanismState) -> None:
        """State mutation hook fired right before the environment steps forward."""
        pass

    def get_solvers(self) -> list[Solver]:
        """Inject native inspect_ai Solvers into the generation pipeline."""
        return []

    def hides_history(self) -> bool:
        """If True, the orchestrator hides the full chat history from the generation context."""
        return False
