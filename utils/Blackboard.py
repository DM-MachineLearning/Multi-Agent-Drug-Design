import logging
from collections import deque

from utils.utils import extract_property_keys, load_property_config
PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Blackboard:
    """
    This class implements a blackboard system for managing tasks related to fixing molecular properties.
    It maintains separate queues for each property that needs to be fixed and a hall of fame for successful molecules.

    Attributes:
        task_queues (dict): A dictionary mapping property tags to their respective task queues.
        hall_of_fame (list): A list to store successful molecules and their scores.

    Methods:
        post_task(property_to_fix, z, current_scores): Posts a new task to the appropriate property queue.
        fetch_task(property_to_fix): Fetches a task from the specified property queue.
        post_to_hall_of_fame(z, scores): Adds a successful molecule to the hall of fame.
    """
    def __init__(self, config):
        """
        Initializes the Blackboard with task queues for each property defined in the configuration.

        Args:
            config (dict): Configuration dictionary containing property definitions.

        Updates:
            task_queues (dict): Initializes empty deques for each property that needs fixing.
            hall_of_fame (list): Initializes an empty list for successful molecules.
        """
        all_properties = []
        all_properties = extract_property_keys(config)

        self.task_queues = {f"needs_fix:{k}": deque() for k in all_properties}
        self.hall_of_fame = []
        self.z_anchor = None

    def post_task(self, property_to_fix, z, current_scores):
        """
        Posts a new task to the appropriate property queue.

        Args:
            property_to_fix (str): The property that needs to be fixed.
            z: The molecular representation that needs fixing.
            current_scores (dict): The current scores of the molecule.
        
        Updates:
            task_queues (dict): Appends the new task to the corresponding property queue.
        """
        tag = f"needs_fix:{property_to_fix}"
        if tag in self.task_queues:
            self.task_queues[tag].append((z, current_scores))
        else:
            print(f"Warning: No queue found for {property_to_fix}")

    def fetch_task(self, property_to_fix):
        """
        Fetches a task from the specified property queue.

        Args:
            property_to_fix (str): The property for which to fetch a task.
        
        Returns:
            tuple or None: The task (z, current_scores) if available, otherwise None.
        """
        tag = f"needs_fix:{property_to_fix}"
        if tag in self.task_queues and self.task_queues[tag]:
            return self.task_queues[tag].popleft()
        # else:
            # print(f"Warning: No task found for {property_to_fix}. Need to turn back to exploration.")
        return None
    
    def post_to_hall_of_fame(self, z, scores):
        """
        Posts a successful molecule to the hall of fame.

        Args:
            z: The successful molecular representation.
            scores (dict): The scores of the successful molecule.
        
        Updates:
            hall_of_fame (list): Appends the successful molecule and its scores to the hall of fame.
        """
        self.hall_of_fame.append((z, scores))