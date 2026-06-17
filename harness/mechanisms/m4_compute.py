from inspect_ai.solver import Solver, chain_of_thought

from harness.mechanisms.mechanism import Mechanism


class M4ChainOfThought(Mechanism):
    """
    Targets: budget/liveness.
    Enables native inspect_ai chain_of_thought
    """
    def get_solvers(self) -> list[Solver]:
        return [chain_of_thought()]

M4AdaptiveCompute = M4ChainOfThought
