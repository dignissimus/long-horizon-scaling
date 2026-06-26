from environments.textworld import BaseTextWorldExpressEnvironment

class ArithmeticEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "arithmetic" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="arithmetic", step_limit=step_limit)
