from environments.textworld import BaseTextWorldExpressEnvironment

class SortingEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "sorting" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="sorting", step_limit=step_limit)
