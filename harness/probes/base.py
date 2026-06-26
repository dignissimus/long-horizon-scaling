from dataclasses import dataclass, field
from typing import Any
from harness.environment.environment import GameEnvironment

@dataclass(frozen=True)
class ContainerDescription:
    """Safely namespaces containers to eliminate room ambiguity."""
    room_name: str
    container_name: str

@dataclass
class ProbeQuestion:
    """Flexible wrapper allowing strict ground truths or open-ended metadata."""
    id: str                 
    prompt: str             
    metadata: dict[str, Any] = field(default_factory=dict)

class Probe:
    def __init__(self, name: str, interval: int = 1):
        self.name = name
        self.interval = interval

    def should_run(self, step: int) -> bool:
        return step > 0 and step % self.interval == 0

    def before_next_step(self, action: str, obs: str, env: GameEnvironment) -> None:
        """Hook called every step to allow probes to passively track state."""
        pass

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        raise NotImplementedError
