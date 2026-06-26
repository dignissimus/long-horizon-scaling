from environments.textworld import BaseTextWorldExpressEnvironment

class CoinEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "coin" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="coin", step_limit=step_limit)
