from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism

class M8GameRules(Mechanism):
    """
    Targets: LLM Common Sense / Semantic Association.
    Explicitly provides the agent with the internal environment rules for mapping
    cooking preparation verbs (fry, roast, grill) to specific appliances, and 
    reinforces the strictness of cutting mechanics.
    """
    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        rules = env.format_rules()
        if not rules:
            return current_prompt
            
        semantic_prompt = f"\n--- [M8 Game Rules] ---\n{rules}\n"
        return current_prompt + semantic_prompt

    def format_probe_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        return current_prompt
