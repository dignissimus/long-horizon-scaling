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
        semantic_prompt = (
            f"\n--- [M8 Game Rules] ---\n"
            f"To successfully follow recipes, you must use the correct appliance or tool for the required preparation method. Use the following explicit mappings:\n\n"
            f"Cooking Appliances:\n"
            f"- To FRY an ingredient: Use the Stove.\n"
            f"- To ROAST an ingredient: Use the Oven.\n"
            f"- To GRILL an ingredient: Use the Barbeque.\n\n"
            f"Cutting Techniques:\n"
            f"- You must exactly match the cutting verb requested by the recipe (chop, slice, or dice).\n"
            f"- Do not substitute one cutting method for another. If a recipe asks for a 'sliced' apple, you must 'slice apple with knife', not 'chop' or 'dice' it.\n\n"
            f"Ingredient Locations:\n"
            f"- Dry goods (flour, sugar, salt, pepper, oil) are usually on the pantry shelf.\n"
            f"- Cold items (milk, cheese, meats) are usually in the fridge.\n"
            f"- Fresh produce (carrots, tomatoes, apples, onions) are usually in the garden."
        )
        return current_prompt + semantic_prompt

    def format_probe_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        return current_prompt
