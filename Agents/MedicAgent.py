import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from Agents.BaseAgent import BaseAgent

class MedicAgent(BaseAgent):
    """
    MedicAgent specializes in optimizing molecules for a specific property using gradient ascent.
    It retrieves tasks from the blackboard, performs optimization, and routes the results back.

    Attributes:
        specialty (str): The property that the MedicAgent specializes in optimizing.

    Methods:
        run_step(): Retrieves a task from the blackboard, optimizes the molecule if a task is available,
                     or generates a random molecule if no task is present, and analyzes the result.
    """
    def __init__(self, specialty_property, vae, engine, board):
        """
        Initializes the MedicAgent with its specialty property, VAE model, scoring engine, and blackboard.

        Args:
            - specialty_property (str): The property to optimize (e.g., 'solubility').
            - vae (VAE): The shared Variational Autoencoder model.
            - scoring_engine (ScoringEngine): The shared scoring engine.
            - blackboard (Blackboard): The shared blackboard.
        """
        super().__init__(specialty_property, vae, engine, board)
        self.specialty = specialty_property

    def run_step(self):
        """
        Retrieves a task from the blackboard, optimizes the molecule if a task is available,
        or generates a random molecule if no task is present, and analyzes the result.
        """
        task = self.board.fetch_task(self.specialty)

        if task:
            flaw_prop, z_start, _ = task
            logger.info(f"Medic {self.specialty}: [FIXING] Optimization for {flaw_prop}")

            z_out = self.gradient_ascent(
                z_start, 
                self.specialty,
                steps=20, 
                lr=0.05, 
                constraint_z=z_start,
                lambda_penalty=10.0
            )
        else:
            logger.info(f"Medic {self.specialty}: [IDLE] Switching to Hunter Mode...")
            z_out = self.vae.generate_molecule()

        self.analyze_and_route(z_out)