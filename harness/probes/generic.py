from harness.environment.environment import GameEnvironment
from harness.probes.base import Probe, ProbeQuestion

class GenericDriftProbe(Probe):
    def __init__(self, interval: int = 1):
        super().__init__(name="drift", interval=interval)

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        return [ProbeQuestion(
            id="drift_goal",
            prompt=(
                "Based on your history, output your current intentions in exactly this format with nothing else:\n"
                "Goal: ...\n"
                "Current subgoal: ...\n"
                "Planned subsequent subgoals: ..."
            ),
            metadata={"game_goal": getattr(env, 'last_task_desc', '')}
        )]

class GenericIntegrationProbe(Probe):
    def __init__(self, interval: int = 1):
        super().__init__(name="integration", interval=interval)

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        return [ProbeQuestion(
            id="integration_plan",
            prompt=(
                "Based on your goal and current state, write a step-by-step plan for the next 3 actions. "
                "Respond ONLY with the plan and no conversational filler."
            ),
            metadata={}
        )]
