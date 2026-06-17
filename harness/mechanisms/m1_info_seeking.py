from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism


# TODO: I want to think about the implementation of this. Might be fine as is
class M1InfoSeeking(Mechanism):
    """
    Targets: accepted-local-error, drift.
    Injects instructions forcing the agent to evaluate its assumptions and issue 
    information-gathering commands (look, inventory) before interacting with the world.
    """
    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        raise NotImplementedError
        m1_instruction = (
            "\n--- [M1 Verification Policy] ---\n"
            "Before acting on a belief, verify it. If you are making assumptions about your "
            "location or inventory, output ONLY 'look' or 'inventory' to check the world first."
        )
        return current_prompt + m1_instruction
