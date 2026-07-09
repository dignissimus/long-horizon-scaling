from environments.textworld import BaseTextWorldExpressEnvironment

class TextWorldCommonSenseEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "twc" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="twc", step_limit=step_limit)

    def format_rules(self) -> str:
        return (
            "--- DOMAIN KNOWLEDGE ---\n"
            "Real-world household semantic mappings:\n"
            "- Perishable food items (e.g., apple, milk, cheese) are stored in the fridge.\n"
            "- Clean clothing items (e.g., shirt, pants, hat) are stored in a wardrobe or chest of drawers.\n"
            "- Dirty clothing items belong in the washing machine or laundry basket.\n"
            "- Cutlery and dishware (e.g., fork, plate) belong in the dishwasher or kitchen cabinet.\n"
            "- Toiletries (e.g., soap, toothbrush) belong in the bathroom cabinet."
        )
