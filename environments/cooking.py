from environments.textworld import BaseTextWorldExpressEnvironment

class CookingWorldEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "cookingworld" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="cookingworld", step_limit=step_limit)
