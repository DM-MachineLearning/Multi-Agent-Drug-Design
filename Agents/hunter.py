import torch

from BaseAgent import BaseAgent

from Generators.VAE import VAE

from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine

class HunterAgent(BaseAgent):
    """Explores random space to find high-activity leads."""
    def __init__(self, agent_id, vae_backbone: VAE, scoring_engine: ScoringEngine, blackboard: Blackboard):
        super().__init__(agent_id, vae_backbone, scoring_engine, blackboard)

    def run_step(self):
        # 1. Random Sample
        z = self.vae.generate_molecule(batch_size=1)
        
        # 2. Optimize for Activity (No constraints)
        z_optimized = self.gradient_ascent(z, 'activity')
        
        # 3. Analyze results
        self.analyze_and_route(z_optimized)