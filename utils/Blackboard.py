from collections import deque

from utils import extract_property_keys, load_property_config
PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")

class Blackboard:
    """The Shared Memory."""
    def __init__(self, config):
        all_properties = []
        all_properties = extract_property_keys(config)

        self.task_queues = {f"needs_fix:{k}": deque() for k in all_properties}
        self.hall_of_fame = []

    def post_task(self, property_to_fix, z, current_scores):
        tag = f"needs_fix:{property_to_fix}"
        if tag in self.task_queues:
            self.task_queues[tag].append((z, current_scores))
        else:
            print(f"Warning: No queue found for {property_to_fix}")

    def fetch_task(self, property_to_fix):
        tag = f"needs_fix:{property_to_fix}"
        if tag in self.task_queues and self.task_queues[tag]:
            return self.task_queues[tag].popleft()
        else:
            print(f"Warning: No task found for {property_to_fix}. Need to turn back to exploration.")
        return None
    
    def post_to_hall_of_fame(self, z, scores):
        self.hall_of_fame.append((z, scores))