from typing import Any, Callable

from inspect_ai.model import ChatMessageUser, ModelOutput, ChatMessageAssistant
from inspect_ai.solver import Generate, TaskState, solver, Solver

from harness.actions.types import MechanismState, RegenerateAction, TakeAction
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism


@solver
def harness_orchestrator(environment_factory: Callable[[], GameEnvironment], mechanisms: list[Mechanism], max_steps: int = 30, seed: int | None = None) -> Solver:
    """
    Main Inspect solver that runs the middleware-style lifecycle loops across a list of mechanisms.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if seed is None:
            raise ValueError("harness_orchestrator requires a 'seed' argument. It cannot be None.")
        
        env: GameEnvironment = environment_factory()
        obs, env_info = env.reset(seed=seed)
        
        mech_state = MechanismState()
        
        # model_kwargs = {"max_tokens": 256}
        model_kwargs = {}
        for m in mechanisms:
            model_kwargs = m.configure_model(model_kwargs)
        
        step_log = []
        for step in range(max_steps):
            base_prompt = f"Step {step+1}/{max_steps}\nCurrent Observation: {obs}\nWhat is your next action? Respond ONLY with your exact action string and nothing else."
            
            for m in mechanisms:
                base_prompt = m.format_prompt(base_prompt, env, mech_state)
            
            for m in mechanisms:
                # TODO: Probably want to rename to make clear that these typically call the model
                base_prompt = await m.augment_prompt_async(base_prompt, env, mech_state, generate)

            # TODO: Do we want to always give the model the full step history?
            # Probably? For consistency
            state.messages.append(ChatMessageUser(content=base_prompt))
            
            for m in mechanisms:
                for s in m.get_solvers():
                    state = await s(state, generate)
            
            response = await generate(state, **model_kwargs)
            action_candidate = response.output.completion.strip("'\" \n\t")
            
            final_action = action_candidate
            for m in mechanisms:
                res = m.after_generation(action_candidate, env, mech_state)
                if isinstance(res, RegenerateAction):
                    # The below implementation is wrong, I want to re-do the env step where the model
                    # generates with the feedback added to the context
                    raise NotImplementedError
                    final_action = f"Clarification: {res.feedback_to_model}"
                    break
                elif isinstance(res, TakeAction):
                    final_action = res.action_string
            
            state.messages.append(ChatMessageAssistant(content=final_action))

            # TODO: Where is this used? I'm wary of storing this in a dictionary
            mech_state.state_cache["last_action"] = final_action
            
            for m in mechanisms:
                m.before_next_step(env, mech_state)
                
            obs, reward, done, step_info = env.step(final_action)
            
            step_log.append({
                "step": step,
                "action_sent": final_action,
                "observation_received": obs,
                "ground_truth_state": env.format_state(),
                "reward": reward
            })

            if done:
                break
                
        # TODO: Should these really be in metadata or somewhere else
        state.metadata["trajectory_telemetry"] = step_log
        state.metadata["final_score"] = env.current_score
        state.output = ModelOutput.from_content(
            "model", 
            "Completed steps with final progress evaluation. Telemetry saved."
        )
        return state

    return solve
