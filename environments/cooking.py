from typing import Any

from textworld_express import TextWorldExpressEnv

from harness.environment.environment import GameEnvironment


class CookingWorldEnvironment(GameEnvironment):
    """
    Wraps the TextWorldExpress 'cookingworld' engine into our standardized GameEnvironment protocol.
    """
    def __init__(self, step_limit: int = 100) -> None:
        self.env = TextWorldExpressEnv(envStepLimit=step_limit)
        self.last_valid_actions = []
        self.last_inventory = ""
        self.last_look = ""
        self.current_score = 0.0

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        env_seed = seed if seed is not None else 42
        obs, info = self.env.reset(seed=env_seed, gameName="cookingworld")
        
        self.last_valid_actions = info.get("validActions", [])
        self.last_inventory = info.get("inventory", "")
        self.last_look = info.get("look", "")
        self.current_score = info.get("score", 0.0)
        
        return obs, info

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        
        self.last_valid_actions = info.get("validActions", [])
        self.last_inventory = info.get("inventory", "")
        self.last_look = info.get("look", "")
        self.current_score = info.get("score", 0.0)
        
        return obs, reward, done, info

    def format_state(self) -> str:
        return f"[System Tracking] Inventory: {self.last_inventory} | Location: {self.last_look}"

    def get_valid_actions(self) -> list[str]:
        return self.last_valid_actions
