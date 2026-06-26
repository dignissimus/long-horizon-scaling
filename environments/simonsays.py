from environments.textworld import BaseTextWorldExpressEnvironment

class SimonSaysEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "simonsays" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="simonsays", step_limit=step_limit)
