from typing import Any

from textworld_express import TextWorldExpressEnv

from harness.environment.environment import GameEnvironment


class CookingWorldEnvironment(GameEnvironment):
    """
    Wraps the TextWorldExpress 'cookingworld' engine into our standardized GameEnvironment protocol.
    """
    def __init__(self) -> None:
        self.env = TextWorldExpressEnv()
        self.last_valid_actions = []
        self.current_score = 0.0

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        env_seed = seed if seed is not None else 42
        obs, info = self.env.reset(seed=env_seed, gameName="cookingworld")
        
        self.last_valid_actions = info.get("validActions", [])
        self.current_score = info.get("score", 0.0)
        
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        
        self.last_valid_actions = info.get("validActions", [])
        self.current_score = info.get("score", 0.0)
        
        return obs, reward, done, info

    def format_state(self) -> str:
        """
        Extracts structural state info. TWE doesn't export a raw JSON graph directly, 
        so we can synthesize it from the valid actions and recent observations, or 
        rely on TWE's internal state tracking if exposed.
        """
        # TODO: Should the agent see the score? I think this is fine
        # TODO: Do subclasses also show the score. This behaviour should be consistent
        return f"[System Tracking] Current Score: {self.current_score} | TWE Step State active."

    def get_valid_actions(self) -> list[str]:
        return self.last_valid_actions
