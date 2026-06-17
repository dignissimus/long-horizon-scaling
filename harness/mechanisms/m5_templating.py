from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism


# TODO: Doesn't actually force the agent to output valid actions. But I think this is fine
class M5ActionTemplating(Mechanism):
    """
    Targets: interface.
    Restricts the agent exclusively to parser-admissible valid actions.
    """
    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        valid_actions = env.get_valid_actions()
        if valid_actions:
            actions_list = ", ".join(f"'{a}'" for a in valid_actions)
            constraint = (
                f"\n--- [M5 Admissible Actions] ---\n"
                f"You MUST choose your exact action from the following list: {actions_list}\n"
                f"Output ONLY ONE action per turn. Output ONLY the action string. Do NOT add conversational text like 'I will' or any prefixes/suffixes. Do NOT wrap the action in quotes."
            )
            return current_prompt + constraint
        return current_prompt
