from abc import abstractmethod
import torch
import torch.nn.functional as F

from Generators.VAE import VAE

from utils.utils import get_property_details, load_property_config
from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine

PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")

class BaseAgent:
    def __init__(self, agent_property: str, vae_backbone: VAE, scoring_engine: ScoringEngine, blackboard: Blackboard):
        self.agent_property = agent_property
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
        
        # 1. Check Primary Success (Potency)
        potency_cfg = PROPERTY_CONFIG.get('potency')
        if scores['potency'] < potency_cfg['threshold']:
            return # Discard: If it's not potent, we don't fix ADMET yet.

        # 2. Check Hard Filters (Immediate Discard if failed)
        for prop, cfg in PROPERTY_CONFIG.get('hard_filters', {}).items():
            is_bad = (scores[prop] > cfg['threshold']) if cfg['target'] == 'low' else (scores[prop] < cfg['threshold'])
            if is_bad:
                print(f"Agent {self.agent_property}: Discarded due to Hard Filter: {prop}")
                return

        # 3. Check Soft Filters (These are "Fixable" flaws)
        flaws = []
        for prop, cfg in PROPERTY_CONFIG.get('soft_filters', {}).items():
            is_bad = (scores[prop] > cfg['threshold']) if cfg['target'] == 'low' else (scores[prop] < cfg['threshold'])
            if is_bad:
                flaws.append(prop)

        # 4. Routing
        if not flaws:
            self.board.hall_of_fame.append((z, scores))
            print(f"Agent {self.agent_property}: FOUND SUCCESSFUL LEAD!")
        else:
            flaws.sort(key=lambda x: PROPERTY_CONFIG['soft_filters'][x].get('weight', 0), reverse=True)
            primary_flaw = flaws[0]
            
            self.board.post_task(primary_flaw, z, scores)
            print(f"Agent {self.agent_property}: Potent but needs fix for {primary_flaw}. Posted to Board.")

    @abstractmethod
    def run_step(self):
        """Subclasses must implement this."""
        pass