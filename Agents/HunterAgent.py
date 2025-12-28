from Agents.BaseAgent import BaseAgent

class HunterAgent(BaseAgent):
    """
    A HunterAgent is designed to explore the chemical space by generating random molecular structures
    and optimizing them towards a specified objective property using gradient ascent.
    This is a class derived from BaseAgent.

    Attributes:
        objective (callable): A function that evaluates the desired property of a molecule.
    
    Methods:
        run_step(): Generates a random molecule, optimizes it using gradient ascent towards the objective,
                     and analyzes the optimized molecule.
    """
    def __init__(self, objective_prop, vae, engine, board):
        """
        Initializes the HunterAgent with its specific objective property, VAE model, scoring engine, and blackboard.

        Args:
            - objective_prop (str): The property to optimize (e.g., 'potency').
            - vae (VAE): The shared Variational Autoencoder model.
            - scoring_engine (ScoringEngine): The shared scoring engine.
            - blackboard (Blackboard): The shared blackboard.
        """
        super().__init__(objective_prop, vae, engine, board)
        self.objective = objective_prop

    def run_step(self):
        """
        Executes a single step of the HunterAgent's behavior:
        1. Generates a random molecule in latent space.
        2. Optimizes the molecule using gradient ascent towards the agent's objective property.
        3. Analyzes the optimized molecule and routes it accordingly.
        """
        token_ids, z_init = self.vae.generate_molecule()
        z_optimized = self.gradient_ascent(
            z_init,
            self.objective,
            steps=50,
            lr=0.1, 
            constraint_z=None,
            lambda_penalty=0
        )
        self.analyze_and_route(z_optimized)