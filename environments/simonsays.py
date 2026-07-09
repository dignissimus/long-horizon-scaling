from environments.textworld import BaseTextWorldExpressEnvironment

class SimonSaysEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "simonsays" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="simonsays", step_limit=step_limit)

    def format_rules(self) -> str:
        return (
            "--- DOMAIN KNOWLEDGE ---\n"
            "The rules of Simon Says:\n"
            "- You must only perform an action if the instruction is explicitly prefaced with 'Simon says, ...'\n"
            "- If the instruction does NOT begin with 'Simon says', it is a trick and you must not execute that action."
        )
