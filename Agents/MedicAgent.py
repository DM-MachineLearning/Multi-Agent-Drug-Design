import torch

from BaseAgent import BaseAgent

from Generators.VAE import VAE

from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine

class MedicAgent(BaseAgent):
    """Fixes specific properties."""
    def __init__(self, agent_id, vae: VAE, engine: ScoringEngine, board: Blackboard, specialty_property):
        super().__init__(agent_id, vae, engine, board)
        self.specialty = specialty_property # e.g., 'toxicity'

    def run_step(self):
        # 1. Look for work
        task = self.board.fetch_task(self.specialty)
        
        if task:
            z_start, _ = task
            # 2. Fix the specific flaw (WITH constraints)
            z_fixed = self.gradient_ascent(
                z_start, 
                self.specialty, 
                constraint_z=z_start, # Crucial: Don't lose the Activity!
                lambda_penalty=5.0
            )
            # 3. Analyze (Did I fix it? Did I break Activity?)
            self.analyze_and_route(z_fixed)
        else:
            # Improvisation: If no work, act like a Hunter for a bit!
            # print("Medic bored. Hunting...")
            super().gradient_ascent(torch.randn(1, 128), 'activity')