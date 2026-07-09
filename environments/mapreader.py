from environments.textworld import BaseTextWorldExpressEnvironment

class MapReaderEnvironment(BaseTextWorldExpressEnvironment):
    """
    Wraps the TextWorldExpress "mapreader" engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        super().__init__(game_name="mapreader", step_limit=step_limit)

    def format_rules(self) -> str:
        return (
            "--- DOMAIN KNOWLEDGE ---\n"
            "Navigation and Maps:\n"
            "- A map provides a topological layout of the environment.\n"
            "- Reading a map reveals the spatial connectivity between rooms via cardinal directions (North, South, East, West).\n"
            "- You can navigate to adjacent connected rooms by issuing the corresponding compass direction."
        )
