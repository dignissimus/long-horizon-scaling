from typing import Any, Protocol


class GameEnvironment(Protocol):
    """Protocol enforcing structural compatibility for text-based long-horizon environments."""
    
    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """Resets the environment and returns the initial observation and metadata."""
        ...

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Executes an action. Returns (observation, reward, done, info_dict)."""
        ...

    def format_state(self) -> str:
        """Returns an explicitly structured text representation of the current true state."""
        ...

    def get_valid_actions(self) -> list[str]:
        """Returns a list of admissible actions for the current state (used by M5)."""
        ...

    # Current tracked score of the environment run
    current_score: float
