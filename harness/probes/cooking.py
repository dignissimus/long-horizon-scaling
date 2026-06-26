import json
import re
from harness.environment.environment import GameEnvironment
from harness.probes.base import Probe, ProbeQuestion, ContainerDescription

class CookingALEProbe(Probe):
    def __init__(self, interval: int = 1):
        super().__init__(name="ale", interval=interval)
        self.seen_rooms: set[str] = set()
        self.seen_containers: set[ContainerDescription] = set()

    def before_next_step(self, action: str, obs: str, env: GameEnvironment) -> None:
        """Passive tracking of seen rooms and opened containers."""
        if not hasattr(env, 'last_look') or not env.last_look:
            return
            
        room_match = re.search(r"You are in the (.*?)\.", env.last_look)
        if room_match:
            current_room = room_match.group(1).strip()
            self.seen_rooms.add(current_room)
            tree = env.get_object_tree()
            room_data = tree.get("locations", {}).get(current_room, {})
            contents_data = room_data.get("contents", {})
            for item_name, item_data in contents_data.items():
                if isinstance(item_data, dict) and item_data.get("isContainer") and item_data.get("isOpen"):
                    self.seen_containers.add(ContainerDescription(room_name=current_room, container_name=item_name))

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        questions = []
        
        # Get ground truth from the engine's serialization tree
        tree = env.get_object_tree()

        # 1. Inventory Probe
        inventory_items = list(tree.get("inventory", {}).keys())
        questions.append(ProbeQuestion(
            id="inventory",
            prompt=(
                "Based on your history, list the items currently in your inventory. "
                "Output as a line-separated list containing nothing else. Do not include conversational filler. "
                "Example output:\nitem one\nitem two"
            ),
            metadata={"ground_truth": inventory_items}
        ))
        
        # 2. Room Probes
        locations_data = tree.get("locations", {})
        for room in self.seen_rooms:
            room_data = locations_data.get(room, {})
            contents_data = room_data.get("contents", {})
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
                    "Example output:\nitem one\nitem two"
                ),
                metadata={"ground_truth": visible_items}
            ))
            
        # 3. Container Probes
        for container in self.seen_containers:
            room_data = locations_data.get(container.room_name, {})
            room_contents = room_data.get("contents", {})
            container_data = room_contents.get(container.container_name, {})
            
            contents = []
            if container_data and isinstance(container_data, dict):
                contents_dict = container_data.get("contents", {})
                contents = list(contents_dict.keys())
                
            questions.append(ProbeQuestion(
                id=f"container_{container.room_name}_{container.container_name}",
                prompt=(
                    f"Based on your history, list the items inside the {container.container_name} located in the {container.room_name}. "
                    "Output as a line-separated list containing nothing else. Do not include conversational filler. "
                    "Example output:\nitem one\nitem two"
                ),
                metadata={"ground_truth": contents}
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
