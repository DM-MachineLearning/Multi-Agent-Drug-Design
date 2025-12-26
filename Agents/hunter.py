from BaseAgent import BaseAgent

class HunterAgent(BaseAgent):
    def __init__(self, agent_property, vae, engine, board, objective_prop="potency"):
        super().__init__(agent_property, vae, engine, board)
        self.objective = objective_prop

    def run_step(self):
        z_random = self.vae.generate_molecule()

        print(f"Hunter {self.agent_property}: Hunting for new {self.objective} leads...")
        z_optimized = self.gradient_ascent(
            z_random, 
            self.objective, 
            steps=50,
            lr=0.1, 
            constraint_z=None,
            lambda_penalty=0
        )
        
        self.analyze_and_route(z_optimized)