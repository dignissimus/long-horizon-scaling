from environments.textworld import BaseTextWorldExpressEnvironment

class ArithmeticEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "arithmetic" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="arithmetic", step_limit=step_limit)

    def format_rules(self) -> str:
        return (
            "--- DOMAIN KNOWLEDGE ---\n"
            "Mathematical evaluation rules:\n"
            "- Mathematical expressions must be evaluated using the standard Order of Operations (PEMDAS: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)."
        )
