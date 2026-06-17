from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism


class M3StateExternalization(Mechanism):
    """
    Targets: accepted-local-error.
    Surfaces the exact, tracked state directly into the prompt to prevent hallucinated game states.
    """
    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        state_repr = env.format_state()
        return f"{current_prompt}\n--- [M3 Maintained State] ---\n{state_repr}\n"
