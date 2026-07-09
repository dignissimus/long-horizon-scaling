import re
from harness.environment.environment import GameEnvironment
from harness.probes.base import Probe, ProbeQuestion

class NLPInventoryProbe(Probe):
    """Probes the agent on its current inventory using regex parsed from env.last_inventory"""
    def __init__(self, interval: int = 1):
        super().__init__(name="nlp_inventory", interval=interval)

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        inv_str = getattr(env, 'last_inventory', '')
        items = []
        
        # Format 1: Explicit 'inventory' command output -> "You are carrying: a apple, a shirt"
        if "carrying:" in inv_str:
            item_part = inv_str.split("carrying:")[1].strip()
            items = [i.strip().replace("a ", "").replace("an ", "") for i in item_part.split(",")]
        # Format 2: Backend info dict output -> "Inventory:\n  a shaving cream\n  an apple"
        elif "Inventory:" in inv_str and "currently empty" not in inv_str:
            lines = inv_str.split("Inventory:")[1].strip().split("\n")
            items = [line.strip().replace("a ", "").replace("an ", "") for line in lines if line.strip()]

        return [ProbeQuestion(
            id="inventory",
            prompt=(
                "Based on your history, list the items currently in your inventory. "
                "Output as a comma-separated list. If empty, output 'empty'."
            ),
            metadata={"ground_truth": items if items else ["empty"]}
        )]

class NLPRoomContentsProbe(Probe):
    """Probes the agent on what it can currently see in its location using regex parsed from env.last_look"""
    def __init__(self, interval: int = 1):
        super().__init__(name="nlp_room", interval=interval)

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        look_str = getattr(env, 'last_look', '')
        
        # Simple extraction of "You see a X" or "that has X on it"
        items_found = re.findall(r"(?:see a|see an|has) ([\w\s\,]+)(?:on it|\.)", look_str)
        
        # Clean up the regex matches
        clean_items = []
        for match in items_found:
            for piece in match.split(","):
                piece = piece.strip()
                if piece and "nothing" not in piece:
                    piece = piece.replace("a ", "").replace("an ", "").replace("that is closed", "").replace("that is open", "").strip()
                    clean_items.append(piece)

        return [ProbeQuestion(
            id="current_room",
            prompt=(
                "Based on your history, list the interactive objects and items currently visible in your location. "
                "Output as a comma-separated list."
            ),
            metadata={"ground_truth_heuristic": clean_items}
        )]

class MapReaderTopologyProbe(Probe):
    """Probes the agent on the topological locations in MapReader based on the initial task description."""
    def __init__(self, interval: int = 1):
        super().__init__(name="map_topology", interval=interval)

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        task_str = getattr(env, 'last_task_desc', '')
        # e.g. "take the coin that is located in the canteen, and put it into the box found in the steam room"
        coin_room = re.search(r"coin that is located in the ([\w\s]+),", task_str)
        box_room = re.search(r"box found in the ([\w\s]+)\.", task_str)

        coin_loc = coin_room.group(1) if coin_room else "unknown"
        box_loc = box_room.group(1) if box_room else "unknown"

        return [
            ProbeQuestion(
                id="map_coin_location",
                prompt="Based on the task description, which room did the coin start in?",
                metadata={"ground_truth": coin_loc}
            ),
            ProbeQuestion(
                id="map_box_location",
                prompt="Based on the task description, which room is the target box located in?",
                metadata={"ground_truth": box_loc}
            )
        ]

class ArithmeticOperandsProbe(Probe):
    """Probes the agent on the math problem quantities using initial state parsing."""
    def __init__(self, interval: int = 1):
        super().__init__(name="arithmetic_operands", interval=interval)

    def get_questions(self, env: GameEnvironment) -> list[ProbeQuestion]:
        look_str = getattr(env, 'last_look', '')
        # e.g. "patio table that has 13 pineapples, and 64 peaches on it"
        # Since arithmetic is contained in one room, we can just extract all numbers
        numbers = re.findall(r"\b(\d+)\s+([a-zA-Z]+)\b", look_str)
        
        ground_truth = {item: num for num, item in numbers}

        return [ProbeQuestion(
            id="arithmetic_quantities",
            prompt=(
                "Based on your visual history, list the exact quantity of each item in the room. "
                "Format as 'quantity item' (e.g., '13 pineapples')."
            ),
            metadata={"ground_truth": ground_truth}
        )]
