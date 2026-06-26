import json
import re
from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism
from harness.probes.base import ContainerDescription

class M7WorldModelExternalization(Mechanism):
    """
    Like StateExternalization, but provides a persistent world model of everything 
    the agent has discovered so far (rooms and opened containers), 
    extracted from the omniscient JSON tree.
    """
    def __init__(self):
        super().__init__()
        self.seen_rooms: set[str] = set()
        self.seen_containers: set[ContainerDescription] = set()

    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        if not hasattr(env, 'last_look') or not env.last_look:
            return current_prompt
            
        try:
            tree = json.loads(env.get_object_tree())
        except Exception:
            return current_prompt

        room_match = re.search(r"You are in the (.*?)\.", env.last_look)
        if room_match:
            current_room = room_match.group(1).strip()
            self.seen_rooms.add(current_room)
            
            # If any container in the current room is open while we are here, 
            # we feasibly know its contents forever.
            room_data = tree.get("locations", {}).get(current_room, {})
            for item_name, item_data in room_data.items():
                if isinstance(item_data, dict) and item_data.get("isContainer") and item_data.get("isOpen"):
                    self.seen_containers.add(ContainerDescription(room_name=current_room, container_name=item_name))

        # Build the Externalized World Model
        output = ["--- [M7 Known World Model] ---"]
        
        # 1. Inventory
        output.append("Inventory:")
        inventory_items = list(tree.get("inventory", {}).keys())
        if inventory_items:
            for inv_item in inventory_items:
                output.append(f"  - {inv_item}")
        else:
            output.append("  (empty)")

        # 2. Locations
        output.append("Locations Visited:")
        locations_data = tree.get("locations", {})
        for room in sorted(self.seen_rooms):
            output.append(f"  {room}:")
            room_data = locations_data.get(room, {})
            
            visible_items_found = False
            for item_name, item_data in room_data.items():
                if isinstance(item_data, dict) and "name" in item_data:
                    visible_items_found = True
                    output.append(f"    - {item_name}")
                    
                    # 3. Known Containers
                    container_desc = ContainerDescription(room_name=room, container_name=item_name)
                    if container_desc in self.seen_containers:
                        contents = list(item_data.get("contents", {}).keys())
                        if contents:
                            output.append(f"      [Contents: {', '.join(contents)}]")
                        else:
                            output.append(f"      [Contents: empty]")
            
            if not visible_items_found:
                output.append("    (no visible items)")

        return f"{current_prompt}\n" + "\n".join(output) + "\n"
