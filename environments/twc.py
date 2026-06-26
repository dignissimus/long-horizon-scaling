from environments.textworld import BaseTextWorldExpressEnvironment

class TextWorldCommonSenseEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "twc" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="twc", step_limit=step_limit)
