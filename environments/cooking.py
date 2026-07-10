from environments.textworld import BaseTextWorldExpressEnvironment

class CookingWorldEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "cookingworld" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100, game_params: dict = None) -> None:
        super().__init__(game_name="cookingworld", step_limit=step_limit, game_params=game_params)

    def format_rules(self) -> str:
        return (
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

    def format_system_prompt(self) -> str:
        return "You are an autonomous agent playing a text-based game. Your ultimate goal is to explore the environment, locate and read the cookbook, gather the correct ingredients, and prepare the recipe exactly as instructed."
