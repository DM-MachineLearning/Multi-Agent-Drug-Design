from abc import abstractmethod
import torch
import torch.nn.functional as F

from Generators.VAE import VAE

from utils.utils import get_property_details, load_property_config
from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine

PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")

import logging
logger = logging.getLogger(__name__)

class BaseAgent:
    """
    Base class for all Agents in the Multi-Agent Drug Design framework. Contains shared methods and attributes for molecule optimization and evaluation.

    Attributes:
    - agent_property: The property this agent focuses on (e.g., 'potency (activity)', 'toxicity').
    - vae_backbone: The shared VAE model for molecule generation and optimization.
    - scoring_engine: The shared scoring engine for property evaluation.
    - blackboard: The shared blackboard for task posting and retrieval.

    Methods:
    - gradient_ascent: Performs gradient-based optimization in latent space.
    - analyze_and_route: Analyzes a molecule's properties and routes it accordingly.
    - run_step: Abstract method to be implemented by subclasses for agent behavior.
    """
    def __init__(self, agent_property: str, vae_backbone: VAE, scoring_engine: ScoringEngine, blackboard: Blackboard):
        """
        Initializes the BaseAgent with its property focus, VAE backbone, scoring engine, and blackboard.

        Parameters:
        - agent_property (str): The property this agent focuses on.
        - vae_backbone (VAE): The shared VAE model.
        - scoring_engine (ScoringEngine): The shared scoring engine.
        - blackboard (Blackboard): The shared blackboard.
        """
        self.agent_property = agent_property
        self.vae = vae_backbone
        self.scorer = scoring_engine
        self.board = blackboard
        
    def gradient_ascent(self, z, objective_prop, steps=20, lr=0.1, constraint_z=None, lambda_penalty=10.0):
        """
        Performs gradient ascent in the latent space to optimize a molecule for a specific property.
        
        Parameters:
        - z (torch.Tensor): The initial latent vector.
        - objective_prop (str): The property to optimize.
        - steps (int): Number of optimization steps.
        - lr (float): Learning rate for the optimizer.
        - constraint_z (torch.Tensor or None): Optional latent vector to constrain similarity.
        - lambda_penalty (float): Penalty weight for deviation from constraint_z.

        Returns:
        - torch.Tensor: The optimized latent vector.
        """
        z = z.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        target_cfg = get_property_details(PROPERTY_CONFIG, objective_prop)
        if target_cfg is None:
            raise ValueError(f"Property {objective_prop} not found in configuration.")

        maximize = (target_cfg['target'] == 'high') # Determine if we are maximizing or minimizing the property

        for _ in range(steps):
            optimizer.zero_grad()
            score = self.scorer.get_all_scores(z)
            score = score[objective_prop]
            
            loss = -score if maximize else score # Define loss based on optimization direction (maximize or minimize)
            
            # Apply constraint penalty if provided
            if constraint_z is not None:
                dist = F.mse_loss(z, constraint_z)
                loss += lambda_penalty * dist

            loss.backward()
            optimizer.step()

        return z.detach()

    def analyze_and_route(self, z):
        """
        Analyzes the scores of a molecule and decides its fate (Success, Failure, Fixable).
        Firstly, checks primary success (potency). If Potency threshold is not met, the molecule is discarded.
        Then checks hard filters (immediate discard if failed).
        Finally, checks soft filters (fixable flaws). If any soft filter fails, the molecule is posted back to the blackboard for further optimization.

        Parameters:
        - z (torch.Tensor): The latent vector of the molecule to analyze.

        Returns:
        - None

        Changes:
        - Updates the blackboard with successful leads or posts tasks for fixable flaws.
        """
        scores = self.scorer.get_all_scores(z)
        
        # 1. Check Primary Success Criterion (Potency/Activity)
        potency_cfg = PROPERTY_CONFIG.get('potency')
        if scores['potency'] < potency_cfg['threshold']:
            return

        # 2. Check Hard Filters (These are "Non-Negotiable" flaws)
        hard_filter_result = self.check_if_molecule_passes_filters('hard', scores)
        if hard_filter_result is not True:
            logger.warning(f"Agent {self.agent_property}: Discarded due to Hard Filter: {hard_filter_result}")
            return

        # 3. Check Soft Filters (These are "Fixable" flaws)
        flaws = []
        soft_filter_result = self.check_if_molecule_passes_filters('soft', scores)
        if soft_filter_result is not True:
            flaws.append(soft_filter_result)

        # 4. Route Molecule Based on Analysis
        if not flaws:
            self.board.hall_of_fame.append((z, scores))
            logger.info(f"Agent {self.agent_property}: FOUND SUCCESSFUL LEAD!")
        else:
            flaws.sort(key=lambda x: PROPERTY_CONFIG['soft_filters'].get(x, {}).get('weight', 0), reverse=True)
            primary_flaw = flaws[0]
            
            self.board.post_task(primary_flaw, z, scores)
            logger.info(f"Agent {self.agent_property}: Potent but needs fix for {primary_flaw}. Posted to Board.")

    def check_if_molecule_passes_filters(self, type_of_filter: str, scores: dict):
        """
        Checks if a molecule passes all filters of a given type (hard or soft).

        Parameters:
        - type_of_filter (str): 'hard' or 'soft' to specify which filters to check.
        - scores (dict): The property scores of the molecule.

        Returns:
        - Property on which the molecule fails, or True if it passes all filters.
        """
        filter_key = 'hard_filters' if type_of_filter == 'hard' else 'soft_filters'
        filters = PROPERTY_CONFIG.get(filter_key, {})
        
        for prop, cfg in filters.items():
            is_bad = (scores[prop] > cfg['threshold']) if cfg['target'] == 'low' else (scores[prop] < cfg['threshold'])
            if is_bad:
                return prop
        return True
    
    @abstractmethod
    def run_step(self):
        """Subclasses must implement this."""
        pass