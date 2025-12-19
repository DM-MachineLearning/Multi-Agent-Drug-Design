import torch

from BaseAgent import BaseAgent

class HunterAgent(BaseAgent):
    """Explores random space to find high-activity leads."""
    def run_step(self):
        # 1. Random Sample
        z = torch.randn(1, 128) # Assuming 128 dim latent
        
        # 2. Optimize for Activity (No constraints)
        z_optimized = self.gradient_ascent(z, 'activity')
        
        # 3. Analyze results
        self.analyze_and_route(z_optimized)