import json
import re
from harness.actions.types import MechanismState
from harness.environment.environment import GameEnvironment
from harness.mechanisms.mechanism import Mechanism
from harness.probes.base import ContainerDescription
from environments.textworld import BaseTextWorldExpressEnvironment

class M7WorldModelExternalization(Mechanism):
    """
    Like StateExternalization, but provides a persistent world model of everything 
    the agent has discovered so far (rooms and opened containers), 
    extracted from the omniscient JSON tree.
    """
    def __init__(self):
        super().__init__()

    def format_prompt(self, current_prompt: str, env: GameEnvironment, state: MechanismState) -> str:
        if not hasattr(env, 'last_look') or not env.last_look:
            return current_prompt
            
        if not isinstance(env, BaseTextWorldExpressEnvironment):
            raise TypeError("M7WorldModelExternalization requires a BaseTextWorldExpressEnvironment")
            
        tree = env.get_object_tree()
        seen_rooms = env.seen_rooms
        seen_containers = env.seen_containers

        # Build the Externalized World Model
        output = ["--- [M7 Known World Model] ---"]
        
        # 1. Inventory
        output.append("Inventory:")
        inventory_items = list(tree["inventory"].keys())
        if inventory_items:
            for inv_item in inventory_items:
                output.append(f"  - {inv_item}")
        else:
            output.append("  (empty)")

        # 2. Locations
        output.append("--- KNOWN ROOMS ---")
        locations_data = tree["locations"]
        for room in sorted(list(seen_rooms)):
            output.append(f"  {room}:")
            room_data = locations_data[room]
            contents_data = room_data["contents"]
            
            visible_items_found = False
            for item_name, item_data in contents_data.items():
                if isinstance(item_data, dict) and "name" in item_data:
                    visible_items_found = True
                    output.append(f"    - {item_name}")
                    
                    # 3. Known Containers
                    if (room, item_name) in seen_containers:
                        contents = list(item_data["contents"].keys())
                        if contents:
                            output.append(f"      [Contents: {', '.join(contents)}]")
                        else:
                            output.append(f"      [Contents: empty]")
            
            if not visible_items_found:
                output.append("    (no visible items)")

        return f"{current_prompt}\n" + "\n".join(output) + "\n"
