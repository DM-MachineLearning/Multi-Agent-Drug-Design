from abc import abstractmethod
import torch
import torch.nn.functional as F
import logging
import torch
import csv
import os

# --- RDKit Import (Safe) ---
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not found. Validation will be strictly text-based.")

logger = logging.getLogger(__name__)
from Generators.VAE import VAE

from utils.utils import get_property_details, load_property_config, write_successful_molecules_to_csv
from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine

PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")
PATH_CONFIG = load_property_config("configs/paths.yaml")

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_property = agent_property
        self.vae = vae_backbone
        self.scorer = scoring_engine
        self.board = blackboard
        
    # def gradient_ascent(self, z, objective_prop, steps=20, lr=0.1, constraint_z=None, lambda_penalty=10.0):
    #     """
    #     Performs gradient ascent in the latent space to optimize a molecule for a specific property.
        
    #     Parameters:
    #     - z (torch.Tensor): The initial latent vector.
    #     - objective_prop (str): The property to optimize.
    #     - steps (int): Number of optimization steps.
    #     - lr (float): Learning rate for the optimizer.
    #     - constraint_z (torch.Tensor or None): Optional latent vector to constrain similarity.
    #     - lambda_penalty (float): Penalty weight for deviation from constraint_z.

    #     Returns:
    #     - torch.Tensor: The optimized latent vector.
    #     """
    #     z = z.detach().clone().requires_grad_(True)
    #     optimizer = torch.optim.Adam([z], lr=lr)

    #     target_cfg = get_property_details(PROPERTY_CONFIG, objective_prop)
    #     if target_cfg is None:
    #         raise ValueError(f"Property {objective_prop} not found in configuration.")

    #     maximize = (target_cfg['target'] == 'high') # Determine if we are maximizing or minimizing the property

    #     for _ in range(steps):
    #         optimizer.zero_grad()
    #         score = self.scorer.get_all_scores(z)
    #         score = score[objective_prop]
            
    #         loss = -score if maximize else score # Define loss based on optimization direction (maximize or minimize)
            
    #         # Apply constraint penalty if provided
    #         if constraint_z is not None:
    #             dist = F.mse_loss(z, constraint_z)
    #             loss += lambda_penalty * dist

    #         loss.backward()
    #         optimizer.step()

    #     return z.detach()

    def gradient_ascent(self, z, objective_prop, steps=50, lr=0.01, constraint_z=None, lambda_penalty=10.0):
        """
        Optimizes z to maximize/minimize a specific property while keeping it valid.
        """
        # 1. Setup Optimization Variable
        z = z.detach().clone().to(self.device)
        z.requires_grad_(True)
        
        optimizer = torch.optim.Adam([z], lr=lr)

        # 2. Get Property Config (to know if we Minimize or Maximize)
        target_cfg = PROPERTY_CONFIG.get(objective_prop)
        if target_cfg:
            maximize = (target_cfg['target'] == 'high')
        else:
            # Default logic if config is missing
            maximize = True 
            if 'toxicity' in objective_prop.lower() or 'inhibition' in objective_prop.lower():
                maximize = False

        # print(f"   ⚗️ Optimizing {objective_prop}...") 
        print(f"tasks = {self.scorer.admet_classifier_model.task_names}")

        for i in range(steps):
            optimizer.zero_grad()
            
            # --- 3. THE FIX: Access Model via self.scorer ---
            # We check if the property belongs to ADMET or Potency
            
            # Case A: It is an ADMET property (BBBP, Toxicity, etc.)
            if objective_prop in self.scorer.admet_classifier_model.task_names:
                # Call the tensor-returning method on the model inside the scorer
                score = self.scorer.admet_classifier_model.get_task_probability(z, objective_prop)
            
            # Case B: It is Potency/Activity
            elif objective_prop == 'potency':
                # You must ensure your Activity Model also has a tensor method!
                # If not, you might need to implement get_activity_tensor(z)
                score = self.scorer.activity_classifier_model.classify_activity(z)
            
            else:
                # Fallback: Try to get it from generic scores (Risk of crash if not tensor)
                # This matches your "old style" request but acts as a fallback
                all_scores = self.scorer.get_all_scores(z)
                score = all_scores.get(objective_prop)
                if not isinstance(score, torch.Tensor):
                    # If it's a float, we can't optimize. Stop to prevent crash.
                    # print(f"⚠️ Cannot optimize {objective_prop} (No gradient).")
                    break

            # --- 4. Define Loss ---
            # Maximize = Minimize negative score
            task_loss = -score if maximize else score
            
            # --- 5. The "Safety Leash" ---
            # Prevents z from exploding into invalid chemical space
            prior_loss = (z ** 2).mean()
            total_loss = task_loss + (5.0 * prior_loss)

            # --- 6. Constraint Penalty (for Medic Agents) ---
            if constraint_z is not None:
                constraint_z = constraint_z.to(self.device)
                dist_loss = torch.nn.functional.mse_loss(z, constraint_z)
                total_loss += lambda_penalty * dist_loss

            total_loss.backward()
            
            # Clip gradients to prevent teleporting
            torch.nn.utils.clip_grad_norm_([z], 0.1)
            
            optimizer.step()

        return z.detach()

    # def analyze_and_route(self, z):
    #     """
    #     Analyzes the scores of a molecule and decides its fate (Success, Failure, Fixable).
    #     Firstly, checks primary success (potency). If Potency threshold is not met, the molecule is discarded.
    #     Then checks hard filters (immediate discard if failed).
    #     Finally, checks soft filters (fixable flaws). If any soft filter fails, the molecule is posted back to the blackboard for further optimization.

    #     Parameters:
    #     - z (torch.Tensor): The latent vector of the molecule to analyze.

    #     Returns:
    #     - None

    #     Changes:
    #     - Updates the blackboard with successful leads or posts tasks for fixable flaws.
    #     """
    #     scores = self.scorer.get_all_scores(z)
        
    #     # 1. Check Primary Success Criterion (Potency/Activity)
    #     potency_cfg = PROPERTY_CONFIG.get('potency')
    #     if scores['potency'] < potency_cfg['threshold']:
    #         return

    #     # 2. Check Hard Filters (These are "Non-Negotiable" flaws)
    #     hard_filter_result = self.check_if_molecule_passes_filters('hard', scores)
    #     if hard_filter_result is not True:
    #         logger.warning(f"Agent {self.agent_property}: Discarded due to Hard Filter: {hard_filter_result}")
    #         return

    #     # 3. Check Soft Filters (These are "Fixable" flaws)
    #     flaws = []
    #     soft_filter_result = self.check_if_molecule_passes_filters('soft', scores)
    #     if soft_filter_result is not True:
    #         flaws.append(soft_filter_result)

    #     # 4. Route Molecule Based on Analysis
    #     if not flaws:
    #         self.board.hall_of_fame.append((z, scores))
    #         logger.info(f"Agent {self.agent_property}: FOUND SUCCESSFUL LEAD! WRTING TO CSV!")
    #         write_successful_molecules_to_csv(
    #             self.board.hall_of_fame, 
    #             PATH_CONFIG['successful_molecules_path'],
    #             vae=self.vae
    #         )
    #     else:
    #         flaws.sort(key=lambda x: PROPERTY_CONFIG['soft_filters'].get(x, {}).get('weight', 0), reverse=True)
    #         primary_flaw = flaws[0]
            
    #         self.board.post_task(primary_flaw, z, scores)
    #         logger.info(f"Agent {self.agent_property}: Potent but needs fix for {primary_flaw}. Posted to Board.")


    def analyze_and_route(self, z):
        """
        Analyzes a latent vector z.
        1. Decodes to SMILES and checks validity (RDKit).
        2. Scores the molecule.
        3. Checks filters (Potency -> Hard -> Soft).
        4. Routes to Hall of Fame (Success) or Blackboard (Medic Task).
        """
        # --- 1. REALITY CHECK (The "RDKit Wala" Part) ---
        # Before we waste time scoring, let's see if it's a real molecule.
        try:
            # We use generate_molecule(z=z) to decode the vector we found
            smi_check, _ = self.vae.generate_molecule(z=z)
        except Exception as e:
            # If the VAE decoder crashes, it's garbage.
            return 

        # Strict Chemical Validation
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smi_check)
            if mol is None:
                logger.debug(f"Agent {self.agent_property}: Generated invalid SMILES: {smi_check}")
                return # Discard immediately
        else:
            # Fallback if no RDKit: check length and spaces
            if len(smi_check) < 2 or " " in smi_check:
                return

        # --- 2. GET SCORES ---
        # Now that we know it's a valid molecule, we pay the cost to score it.
        scores = self.scorer.get_all_scores(z)

        # --- 3. POTENCY FILTER (Primary Objective) ---
        potency_cfg = PROPERTY_CONFIG.get('potency')
        if scores['potency'] < potency_cfg['threshold']:
            return # Too weak, discard.

        # --- 4. HARD FILTERS (Non-Negotiable) ---
        hard_filter_result = self.check_if_molecule_passes_filters('hard', scores)
        if hard_filter_result is not True:
            # logger.info(f"Agent {self.agent_property}: Failed Hard Filter {hard_filter_result}")
            return # Failed a hard constraint (e.g., hERG toxicity), discard.

        # --- 5. SOFT FILTERS (Fixable Flaws) ---
        # Returns True if perfect, or the name of the property that failed (e.g., 'BBBP')
        flaw_prop = self.check_if_molecule_passes_filters('soft', scores)

        # --- 6. ROUTING ---
        if flaw_prop is True: 
            # ✅ SUCCESS CASE
            # 1. Add to Blackboard Memory
            self.board.hall_of_fame.append((z, scores))
            
            # 2. Log it
            # logger.info(f"Agent {self.agent_property}: 🏆 FOUND SUCCESSFUL LEAD! SMILES: {smi_check}")
            
            # 3. Save to CSV IMMEDIATELY (Safety feature)
            self.write_lead_to_csv(smi_check, scores)
            
        else: 
            # 🚑 MEDIC CASE (Needs Fixing)
            # Post a task: "Here is a potent molecule (z), but it failed 'flaw_prop'. Fix it!"
            self.board.post_task(flaw_prop, z, scores)
            logger.info(f"Agent {self.agent_property}: Potent but needs fix for {flaw_prop}. Posted to Board.")

    def write_lead_to_csv(self, smiles, scores):
        """Helper to write a single lead to CSV immediately."""
        filepath = PATH_CONFIG['successful_molecules_path']
        file_exists = os.path.isfile(filepath)
        
        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                writer.writerow(['smiles', 'captions'])
            
            # Write row
            writer.writerow([smiles, str(scores)])

    # def check_if_molecule_passes_filters(self, type_of_filter: str, scores: dict):
    #     """
    #     Checks filters with different logic for Hard vs Soft:
    #     - HARD: Strict AND logic (Must pass ALL).
    #     - SOFT: Majority logic (Must pass at least 5).
        
    #     Returns:
    #     - True: If it passes.
    #     - Property Name (str): The name of a failed property if it fails.
    #     """
    #     filter_key = 'hard_filters' if type_of_filter == 'hard' else 'soft_filters'
    #     # Safely get the filters dict; default to empty if not found
    #     filters = PROPERTY_CONFIG.get(filter_key, {}) 
        
    #     # --- LOGIC 1: HARD FILTERS (STRICT) ---
    #     if type_of_filter == 'hard':
    #         for prop, cfg in filters.items():
    #             # Check if bad
    #             is_bad = (scores[prop] > cfg['threshold']) if cfg['target'] == 'low' else (scores[prop] < cfg['threshold'])
    #             if is_bad:
    #                 return prop  # Fail immediately on first hard violation
    #         return True

    #     # --- LOGIC 2: SOFT FILTERS (VOTING 5/9) ---
    #     else:
    #         passed_count = 0
    #         failed_props = []

    #         for prop, cfg in filters.items():
    #             # Check if bad
    #             is_bad = (scores[prop] > cfg['threshold']) if cfg['target'] == 'low' else (scores[prop] < cfg['threshold'])
                
    #             if not is_bad:
    #                 passed_count += 1
    #             else:
    #                 failed_props.append(prop)
            
    #         # THE RELAXATION RULE:
    #         # If we passed 5 or more, we consider the molecule "Good Enough"
    #         if passed_count >= 5:
    #             return True
    #         else:
    #             # If we passed fewer than 5, we fail.
    #             # Return the FIRST failed property so the Medic has something specific to fix.
    #             # print(f"Property is {failed_props[0]}, Score is {scores[failed_props[0]]}, threshold is {cfg['threshold']}")
    #             return failed_props[0] if failed_props else True
            
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