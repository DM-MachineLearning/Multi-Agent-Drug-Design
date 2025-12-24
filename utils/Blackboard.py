from collections import deque

from utils.utils import load_property_config
PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")

class Blackboard:
    """The Shared Memory."""
    def __init__(self):
        # Dynamic queues: {'needs_fix:toxicity': [], 'needs_fix:solubility': []}
        self.task_queues = {f"needs_fix:{k}": deque() for k in PROPERTY_CONFIG.keys()}
        self.hall_of_fame = []

    def post_task(self, property_to_fix, z, current_scores):
        tag = f"needs_fix:{property_to_fix}"
        if tag in self.task_queues:
            self.task_queues[tag].append((z, current_scores))

    def fetch_task(self, agent_id):
        """Fetch a task for a specific agent."""
        for key, queue in self.task_queues.items():
            if queue:
                return key.split(":")[1], *queue.popleft()
        return None