from BaseAgent import BaseAgent

class MedicAgent(BaseAgent):
    """
    Priority 1: Fix specific flaws (Constraint Optimization).
    Priority 2: Generate new leads if idle (Pure Exploration).
    """
    def __init__(self, agent_property, vae, engine, board, specialty_property):
        super().__init__(agent_property, vae, engine, board)
        self.specialty = specialty_property

    def run_step(self):
        task = self.board.fetch_task(self.specialty)
        
        if task:
            flaw_prop, z_start, _ = task
            print(f"Medic {self.agent_property}: [FIXING] Optimization for {flaw_prop}")

            z_out = self.gradient_ascent(
                z_start, 
                self.specialty,
                steps=20, 
                lr=0.05, 
                constraint_z=z_start,
                lambda_penalty=10.0
            )
            
        else:
            print(f"Medic {self.agent_property}: [IDLE] Switching to Hunter Mode...")

            z_out = self.vae.generate_molecule()

        self.analyze_and_route(z_out)