import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from Agents.BaseAgent import BaseAgent

class MedicAgent(BaseAgent):
    """
    MedicAgent specializes in fixing specific property flaws (e.g., toxicity, solubility).
    When idle, it supports the Hunter by exploring the neighborhood of the Golden Scaffold.
    """
    def __init__(self, specialty_property, vae, engine, board):
        super().__init__(specialty_property, vae, engine, board)
        self.specialty = specialty_property

    def run_step(self):
        """
        1. Checks for 'fixing' tasks on the blackboard.
        2. If no tasks, generates a new molecule seeded near the Golden Scaffold.
        """
        task = self.board.fetch_task(self.specialty)

        if task:
            # --- TASK MODE: FIXING A FLAWED MOLECULE ---
            z_start, scores = task
            
            # Constraint_z keeps the 'Medic' from changing the molecule too much
            # while it tries to fix the specialty flaw.
            z_out = self.gradient_ascent(
                z_start, 
                self.specialty,
                steps=20, 
                lr=0.05, 
                constraint_z=z_start,
                lambda_penalty=10.0
            )
        else:
            # --- IDLE MODE: SCAFFOLD-SEEDED EXPLORATION ---
            # Instead of purely random generation, check for the search anchor
            if hasattr(self.board, 'z_anchor') and self.board.z_anchor is not None:
                z_center = self.board.z_anchor
                # Medics explore slightly wider than Hunters when idle
                exploration_radius = 0.3 
                noise = torch.randn_like(z_center)
                z_out = z_center + (noise * exploration_radius)
                z_out = z_out.to(self.vae.device)
            else:
                # Fallback to random if main.py hasn't set the anchor yet
                _, z_out = self.vae.generate_molecule()

        # Route the result for scoring and potential Hall of Fame placement
        self.analyze_and_route(z_out)