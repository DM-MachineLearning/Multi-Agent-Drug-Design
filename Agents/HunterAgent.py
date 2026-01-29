# from Agents.BaseAgent import BaseAgent

# class HunterAgent(BaseAgent):
#     """
#     A HunterAgent is designed to explore the chemical space by generating random molecular structures
#     and optimizing them towards a specified objective property using gradient ascent.
#     This is a class derived from BaseAgent.

#     Attributes:
#         objective (callable): A function that evaluates the desired property of a molecule.
    
#     Methods:
#         run_step(): Generates a random molecule, optimizes it using gradient ascent towards the objective,
#                      and analyzes the optimized molecule.
#     """
#     def __init__(self, objective_prop, vae, engine, board):
#         """
#         Initializes the HunterAgent with its specific objective property, VAE model, scoring engine, and blackboard.

#         Args:
#             - objective_prop (str): The property to optimize (e.g., 'potency').
#             - vae (VAE): The shared Variational Autoencoder model.
#             - scoring_engine (ScoringEngine): The shared scoring engine.
#             - blackboard (Blackboard): The shared blackboard.
#         """
#         super().__init__(objective_prop, vae, engine, board)
#         self.objective = objective_prop

#     def run_step(self):
#         """
#         Executes a single step of the HunterAgent's behavior:
#         1. Generates a random molecule in latent space.
#         2. Optimizes the molecule using gradient ascent towards the agent's objective property.
#         3. Analyzes the optimized molecule and routes it accordingly.
#         """
#         token_ids, z_init = self.vae.generate_molecule()
#         z_optimized = self.gradient_ascent(
#             z_init,
#             self.objective,
#             steps=50,
#             lr=0.1, 
#             constraint_z=None,
#             lambda_penalty=0
#         )
#         self.analyze_and_route(z_optimized)

import torch
from Agents.BaseAgent import BaseAgent

class HunterAgent(BaseAgent):
    """
    An upgraded HunterAgent that can perform:
    1. Random Exploration (Global Search)
    2. Scaffold-Seeded Exploration (Local Search around a 'Golden Scaffold')
    """
    def __init__(self, objective_prop, vae, engine, board):
        super().__init__(objective_prop, vae, engine, board)
        self.objective = objective_prop

    def run_step(self):
        """
        Executes a search step. If z_anchor exists on the board, 
        it initializes the search at that coordinate.
        """
        # 1. INITIALIZATION (Scaffold-Seeded vs Random)
        # Check if the 'Golden Scaffold' anchor has been set on the blackboard
        if hasattr(self.board, 'z_anchor') and self.board.z_anchor is not None:
            # SCAFFOLD-SEEDED SEARCH (Exploitation)
            # We add a small amount of Gaussian noise to the anchor 
            # to start at various 'neighbors' of the elite scaffold.
            z_center = self.board.z_anchor
            exploration_radius = 0.2  # Hyperparameter: How far to drift from anchor
            noise = torch.randn_like(z_center)
            z_init = z_center + (noise * exploration_radius)
            
            # Optional: Ensure it's on device
            z_init = z_init.to(self.vae.device)
        else:
            # RANDOM EXPLORATION (Global Search)
            # Fallback if no anchor is provided
            _, z_init = self.vae.generate_molecule()

        # 2. GRADIENT ASCENT (Optimization)
        # We optimize the 'seeded' vector towards the target (e.g., potency)
        z_optimized = self.gradient_ascent(
            z_init,
            self.objective,
            steps=30,     # Lower steps needed when starting from a good scaffold
            lr=0.05,      # Smaller LR to keep the core scaffold structure intact
            constraint_z=None,
            lambda_penalty=0
        )

        # 3. ANALYSIS & ROUTING
        # Decode the optimized vector and send to the board
        self.analyze_and_route(z_optimized)