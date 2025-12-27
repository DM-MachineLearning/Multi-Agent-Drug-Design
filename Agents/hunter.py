from BaseAgent import BaseAgent

class HunterAgent(BaseAgent):
    def __init__(self, objective_prop, vae, engine, board):
        super().__init__(objective_prop, vae, engine, board)
        self.objective = objective_prop

    def run_step(self):
        z_random = self.vae.generate_molecule()
        z_optimized = self.gradient_ascent(
            z_random, 
            self.objective, 
            steps=50,
            lr=0.1, 
            constraint_z=None,
            lambda_penalty=0
        )
        
        self.analyze_and_route(z_optimized)