from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionResult:
    """Base class for all mechanism actions after model generation."""
    pass

@dataclass
class TakeAction(ActionResult):
    """Signals that the action is valid and should be sent to the environment."""
    action_string: str

@dataclass
class RegenerateAction(ActionResult):
    """Signals that the mechanism rejected the action and requests a retry."""
    reason: str
    feedback_to_model: str

# TODO: Should probably remove? Not sure what use this has
@dataclass
class MechanismState:
    """A generic, extensible container for individual mechanism states."""
    memory_scratchpad: str = "No history recorded yet."
    current_plan: str = "No active plan."
    assumptions_log: list[str] = field(default_factory=list)
    state_cache: dict[str, Any] = field(default_factory=dict)
