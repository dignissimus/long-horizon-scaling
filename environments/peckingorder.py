from environments.textworld import BaseTextWorldExpressEnvironment

class PeckingOrderEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "peckingorder" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="peckingorder", step_limit=step_limit)
