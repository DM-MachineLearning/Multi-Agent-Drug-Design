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
        if hasattr(self.board, 'z_anchor') and self.board.z_anchor is not None:
            z_center = self.board.z_anchor
            exploration_radius = 0.2 
            noise = torch.randn_like(z_center)
            z_init = z_center + (noise * exploration_radius)
            
            # Optional: Ensure it's on device
            z_init = z_init.to(self.vae.device)
        else:
            _, z_init = self.vae.generate_molecule()

        # 2. GRADIENT ASCENT (Optimization)
        z_optimized = self.gradient_ascent(
            z_init,
            self.objective,
            steps=30,
            lr=0.05,
            constraint_z=None,
            lambda_penalty=0
        )

        # 3. ANALYSIS & ROUTING
        self.analyze_and_route(z_optimized)