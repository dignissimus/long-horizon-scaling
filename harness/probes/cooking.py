import json
import re
from harness.environment.environment import GameEnvironment
from harness.probes.base import Probe, ProbeQuestion, ContainerDescription
from environments.textworld import BaseTextWorldExpressEnvironment

class CookingALEProbe(Probe):
    def __init__(self, interval: int = 1):
        super().__init__(name="ale", interval=interval)

    def before_next_step(self, action: str, obs: str, env: GameEnvironment) -> None:
        pass

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        questions = []
        
        # Get ground truth from the engine's serialization tree
        tree = env.get_object_tree()

        if not isinstance(env, BaseTextWorldExpressEnvironment):
            raise TypeError("CookingALEProbe requires a BaseTextWorldExpressEnvironment")

        seen_rooms = env.seen_rooms
        seen_containers = env.seen_containers
        
        # 1. Inventory Probe
        inventory_items = list(tree["inventory"].keys())
        questions.append(ProbeQuestion(
            id="inventory",
            prompt=(
                "Based on your history, list the items currently in your inventory. "
                "Output as a line-separated list containing nothing else. Do not include conversational filler. "
                "If there is nothing, output 'nothing'. "
                "Example output:\nitem one\nitem two"
            ),
            metadata={"ground_truth": inventory_items}
        ))
        
        # 2. Room Probes
        locations_data = tree["locations"]
        for room in seen_rooms:
            room_data = locations_data[room]
            contents_data = room_data["contents"]
            visible_items = []
            for item_name, item_info in contents_data.items():
                if isinstance(item_info, dict) and "name" in item_info:
                    visible_items.append(item_name)
                    
            questions.append(ProbeQuestion(
                id=f"room_{room}",
                prompt=(
                    f"Based on your history, list the visible items in the {room}. "
                    "Do not include items inside ANY containers, whether open or closed. "
                    "Output as a line-separated list containing nothing else. Do not include conversational filler. "
                    "If there is nothing, output 'nothing'. "
                    "Example output:\nitem one\nitem two"
                ),
                metadata={"ground_truth": visible_items}
            ))
            
        # 3. Container Probes
        for room_name, container_name in seen_containers:
            room_data = locations_data[room_name]
            room_contents = room_data["contents"]
            container_data = room_contents[container_name]
            
            container_contents = list(container_data["contents"].keys())
            
            questions.append(ProbeQuestion(
                id=f"container_{room_name}_{container_name}",
                prompt=(
                    f"Based on your history, list the items currently inside the {container_name} in the {room_name}. "
                    "Output as a line-separated list containing nothing else. Do not include conversational filler. "
                    "If it is empty or you don't know, output 'nothing'. "
                    "Example output:\nitem one\nitem two"
                ),
                metadata={"ground_truth": container_contents}
            ))
        # 4. Recipe Ingredients Probe
        recipe_data = tree["recipe"]
        recipe_ingredients = [item["name"] for item in recipe_data if "name" in item]
        
        questions.append(ProbeQuestion(
            id="recipe_ingredients",
            prompt=(
                "Based on your history, list the ingredients required by the recipe in the cookbook. "
                "Output as a line-separated list containing nothing else. Do not include conversational filler. "
                "If you have not read the cookbook, output 'nothing'. "
                "Example output:\nitem one\nitem two"
            ),
            metadata={"ground_truth": recipe_ingredients}
        ))
        
        # 5. Recipe Directions Probe
        recipe_directions = []
        for item in recipe_data:
            prep = item.get("preparation", "")
            name = item.get("name", "")
            if not name: continue
            if "sliced" in prep: recipe_directions.append(f"slice the {name}")
            if "chopped" in prep: recipe_directions.append(f"chop the {name}")
            if "diced" in prep: recipe_directions.append(f"dice the {name}")
            if "grilled" in prep: recipe_directions.append(f"grill the {name}")
            if "roasted" in prep: recipe_directions.append(f"roast the {name}")
            if "fried" in prep: recipe_directions.append(f"fry the {name}")
        recipe_directions.append("prepare meal")
        
        questions.append(ProbeQuestion(
            id="recipe_directions",
            prompt=(
                "Based on your history, list the preparation directions required by the recipe in the cookbook. "
                "Output as a line-separated list containing nothing else. Do not include conversational filler. "
                "If you have not read the cookbook, output 'nothing'. "
                "Example output:\nslice the apple\nfry the apple"
            ),
            metadata={"ground_truth": recipe_directions}
        ))
            
        return questions

class CookingDriftProbe(Probe):
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

class CookingIntegrationProbe(Probe):
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
