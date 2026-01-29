# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# from Agents.BaseAgent import BaseAgent

# class MedicAgent(BaseAgent):
#     """
#     MedicAgent specializes in optimizing molecules for a specific property using gradient ascent.
#     It retrieves tasks from the blackboard, performs optimization, and routes the results back.

#     Attributes:
#         specialty (str): The property that the MedicAgent specializes in optimizing.

#     Methods:
#         run_step(): Retrieves a task from the blackboard, optimizes the molecule if a task is available,
#                      or generates a random molecule if no task is present, and analyzes the result.
#     """
#     def __init__(self, specialty_property, vae, engine, board):
#         """
#         Initializes the MedicAgent with its specialty property, VAE model, scoring engine, and blackboard.

#         Args:
#             - specialty_property (str): The property to optimize (e.g., 'solubility').
#             - vae (VAE): The shared Variational Autoencoder model.
#             - scoring_engine (ScoringEngine): The shared scoring engine.
#             - blackboard (Blackboard): The shared blackboard.
#         """
#         super().__init__(specialty_property, vae, engine, board)
#         self.specialty = specialty_property

#     # def run_step(self):
#     #     """
#     #     Retrieves a task from the blackboard, optimizes the molecule if a task is available,
#     #     or generates a random molecule if no task is present, and analyzes the result.
#     #     """
#     #     task = self.board.fetch_task(self.specialty)

#     #     if task:
#     #         flaw_prop, z_start, _ = task
#     #         logger.info(f"Medic {self.specialty}: [FIXING] Optimization for {flaw_prop}")

#     #         z_out = self.gradient_ascent(
#     #             z_start, 
#     #             self.specialty,
#     #             steps=20, 
#     #             lr=0.05, 
#     #             constraint_z=z_start,
#     #             lambda_penalty=10.0
#     #         )
#     #     else:
#     #         logger.info(f"Medic {self.specialty}: [IDLE] Switching to Hunter Mode...")
#     #         z_out = self.vae.generate_molecule()

#     #     self.analyze_and_route(z_out)

#     def run_step(self):
#         """
#         Retrieves a task from the blackboard, optimizes the molecule if a task is available,
#         or generates a random molecule if no task is present, and analyzes the result.
#         """
#         task = self.board.fetch_task(self.specialty)

#         if task:
#             z_start, scores = task
#             flaw_prop = self.specialty
#             # logger.info(f"Medic {self.specialty}: [FIXING] Optimization for {flaw_prop}")

#             z_out = self.gradient_ascent(
#                 z_start, 
#                 self.specialty,
#                 steps=20, 
#                 lr=0.05, 
#                 constraint_z=z_start,
#                 lambda_penalty=10.0
#             )
#         else:
#             # logger.info(f"Medic {self.specialty}: [IDLE] Switching to Hunter Mode...")
            
#             # --- MINIMAL CHANGE HERE ---
#             # Unpack the tuple. We ignore the SMILES (first item) and keep z (second item).
#             _, z_out = self.vae.generate_molecule()
            
#             # Optional: You might want to run a quick optimization here too, like a Hunter,
#             # but for purely minimal changes to fix the crash, this is all you need.

#         self.analyze_and_route(z_out)

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