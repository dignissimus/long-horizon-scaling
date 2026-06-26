from environments.textworld import BaseTextWorldExpressEnvironment

class MapReaderEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "mapreader" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="mapreader", step_limit=step_limit)
