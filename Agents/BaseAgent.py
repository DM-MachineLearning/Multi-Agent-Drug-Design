import torch
import torch.nn.functional as F

from Generators.VAE import VAE

from utils.utils import get_property_details, load_property_config
from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine

PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")

class BaseAgent:
    def __init__(self, agent_property: str, vae_backbone: VAE, scoring_engine: ScoringEngine, blackboard: Blackboard):
        self.id = agent_property
        self.vae = vae_backbone
        self.scorer = scoring_engine
        self.board = blackboard
        
    def gradient_ascent(self, z, objective_prop, steps=20, lr=0.1, constraint_z=None, lambda_penalty=10.0):
        """
        The Universal Optimization Function.
        - If constraint_z is None: Pure exploration (Hunter).
        - If constraint_z is set: Constrained optimization (Medic).
        """
        z = z.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        
        target_cfg = get_property_details(PROPERTY_CONFIG, objective_prop)
        if target_cfg is None:
            raise ValueError(f"Property {objective_prop} not found in configuration.")

        maximize = (target_cfg['target'] == 'high')

        for _ in range(steps):
            optimizer.zero_grad()
            score = self.scorer.get_score(z, objective_prop)
            
            # 1. Base Loss (Maximize or Minimize property)
            loss = -score if maximize else score
            
            # 2. Constraint Penalty (Keep structure similar)
            if constraint_z is not None:
                dist = F.mse_loss(z, constraint_z)
                loss += lambda_penalty * dist
            
            loss.backward()
            optimizer.step()
            
        return z.detach()

    def analyze_and_route(self, z):
        """Decides if a molecule is a Success, a Failure, or a 'Fixable' task."""
        scores = self.scorer.get_all_scores(z)
        
        # Check primary success (e.g., is it active?)
        if scores['activity'] < PROPERTY_CONFIG['activity']['threshold']:
            return # Discard, not active enough to care about
            
        # If active, check for flaws (ADMET issues)
        flaws = []
        for prop, cfg in PROPERTY_CONFIG.items():
            if prop == 'activity': continue
            
            is_bad = (scores[prop] > cfg['threshold']) if cfg['target'] == 'low' else (scores[prop] < cfg['threshold'])
            if is_bad:
                flaws.append(prop)
        
        if not flaws:
            self.board.hall_of_fame.append((z, scores))
            print(f"Agent {self.id}: FOUND PERFECT MOLECULE!")
        else:
            # Post the first major flaw to the blackboard
            primary_flaw = flaws[0] # Priorities can be set here
            self.board.post_task(primary_flaw, z, scores)
            print(f"Agent {self.id}: Active but bad {primary_flaw}. Posted to Board.")

    def run_step(self):
        """One step of the agent's operation: Generate/Fix and Analyze."""
        task = self.board.fetch_task(self.agent_property)
        
        if task is None:
            # No task assigned, pure exploration
            z = torch.randn((1, self.vae.latent_dim)) # Sample random latent vector
            print(f"Agent {self.id}: Exploring new molecule.")
        else:
            # Task assigned, attempt to fix
            flaw_prop, constraint_z, _ = task
            print(f"Agent {self.id}: Fixing molecule for {flaw_prop}.")
            z = self.gradient_ascent(constraint_z, flaw_prop, constraint_z=constraint_z)
        
        # Analyze the generated/fixed molecule
        self.analyze_and_route(z)