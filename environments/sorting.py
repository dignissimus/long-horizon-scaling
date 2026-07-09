from environments.textworld import BaseTextWorldExpressEnvironment

class SortingEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "sorting" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="sorting", step_limit=step_limit)

    def format_rules(self) -> str:
        return (
            "--- DOMAIN KNOWLEDGE ---\n"
            "Metric unit conversions to remember:\n"
            "- 1 kg (kilogram) = 1,000 g (grams) = 1,000,000 mg (milligrams).\n"
            "- 1 g (gram) = 1,000 mg (milligrams).\n"
            "Always normalize quantities to the same unit (e.g., milligrams) before comparing them."
        )
