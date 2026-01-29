# import logging
# import torch

# from src.madm.properties.multitask_mlp import MultiTaskADMETModel

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ADMETClassifier:
#     """
#     A classifier for predicting ADMET properties using a pre-trained Multi-Task ADMET model.
#     The model is loaded from the specified path and can classify ADMET properties for given SMILES strings.
#     The model consists of a shared trunk and multiple task-specific heads. It predicts 11 properties such
#     as BBBP, Caco2 permeability, CYP inhibition, HERG inhibition, HLM stability, P-gp substrate, and RLM stability.

#     Attributes:
#         model_path (str): Path to the pre-trained model weights.
#         task_names (list): List of ADMET property names.
#         input_dim (int): Dimension of the input features.
#         hidden_dims (list): List of hidden layer dimensions for the shared trunk.
#         model (MultiTaskADMETModel): The loaded Multi-Task ADMET model.

#     Methods:
#         classify_admet(smiles: str) -> dict:
#             Classify ADMET properties for a given SMILES string and return predicted probabilities.
#     """
#     def __init__(self, model_path: str):
#         """
#         Initialize the ADMETClassifier by loading the pre-trained model.
#         """
#         self.model_path = model_path
#         self.task_names = [
#             'BBBP', 'Caco2_permeability', 'CYP1A2_inhibition', 'CYP2C9_inhibition', 
#             'CYP2C19_inhibition', 'CYP2D6_inhibition', 'CYP3A4_inhibition', 
#             'HERG_inhibition', 'HLM_stability', 'P-gp_substrate', 'RLM_stability'
#         ]
#         self.input_dim = 2055
#         self.hidden_dims = [512, 256] 

#         self.model = MultiTaskADMETModel(
#             input_dim=self.input_dim,
#             shared_hidden_dims=self.hidden_dims,
#             task_names=self.task_names
#         )

#         # Load and Remap State Dict
#         self.state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
#         self.new_state_dict = {}
#         for key, value in self.state_dict.items():
#             if key.startswith("trunk."):
#                 new_key = key.replace("trunk.", "trunk.network.", 1)
#                 self.new_state_dict[new_key] = value
#             else:
#                 self.new_state_dict[key] = value

#         self.model.load_state_dict(self.new_state_dict)
#         self.model.eval()
#         logger.info(f"Model weights loaded successfully with INPUT_DIM={self.input_dim}.")

#     def classify_admet(self, smiles) -> dict:
#         """
#         Classify ADMET properties for a given SMILES string.

#         Args:
#             smiles (str): The SMILES representation of the molecule.
#         Returns:
#             dict: A dictionary with task names as keys and predicted probabilities as values.
#         """
#         with torch.no_grad():
#             logits_dict = self.model(smiles)

#         probabilities = {}
#         for task, logits in logits_dict.items():
#             prob = torch.sigmoid(logits).item()
#             probabilities[task] = prob

#         return probabilities
    
# # --------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------------

# """
# The Below code is just an example of how to use the above class to load the model. 
# It is commented out to avoid execution during import.
# """

# # 1. Configuration - Updated to match the checkpoint shape
# # TASK_NAMES = [
# #     'BBBP', 'Caco2_permeability', 'CYP1A2_inhibition', 'CYP2C9_inhibition', 
# #     'CYP2C19_inhibition', 'CYP2D6_inhibition', 'CYP3A4_inhibition', 
# #     'HERG_inhibition', 'HLM_stability', 'P-gp_substrate', 'RLM_stability'
# # ]
# # INPUT_DIM = 2055  # Changed from 2048 to 2055 based on the error
# # HIDDEN_DIMS = [512, 256] 
# # MODEL_PATH = 'outputs_multitask/multitask_model.pt'

# # # 2. Initialize the model
# # model = MultiTaskADMETModel(
# #     input_dim=INPUT_DIM,
# #     shared_hidden_dims=HIDDEN_DIMS,
# #     task_names=TASK_NAMES
# # )

# # 3. Load and Remap State Dict
# # try:
# #     state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
# #     # Map the old keys to the new class structure
# #     new_state_dict = {}
# #     for key, value in state_dict.items():
# #         if key.startswith("trunk."):
# #             new_key = key.replace("trunk.", "trunk.network.", 1)
# #             new_state_dict[new_key] = value
# #         else:
# #             new_state_dict[key] = value

# #     model.load_state_dict(new_state_dict)
# #     model.eval()
# #     print("Model weights loaded successfully with INPUT_DIM=2055.")
# # except Exception as e:
# #     print(f"Loading failed: {e}")
# #     exit()


import logging
import torch

from Models.AdmetClassifier.train_multitask import MultiHeadADMET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADMETClassifier:
    """
    Updated Classifier for predicting ADMET properties directly from 
    the VAE Latent Space (z).
    """
    def __init__(self, model_path: str, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # EXACT Task Order from your successful training log
        self.task_names = [
            'BBBP', 'CYP1A2_inhibition', 'CYP2C19_inhibition', 'CYP2C9_inhibition', 
            'CYP2D6_inhibition', 'CYP3A4_inhibition', 'Caco2_permeability', 
            'HLM_stability', 'P-gp_substrate', 'RLM_stability', 'hERG_inhibition'
        ]
        
        self.input_dim = 128 # NEW: Latent space dim, not fingerprints
        self.model = MultiHeadADMET(latent_dim=self.input_dim, num_tasks=len(self.task_names))

        # Load Checkpoint
        logger.info(f"🚀 Loading Latent-ADMET model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle the 'model_state' key from our training script
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device).eval()
        logger.info(f"✅ ADMET Classifier Ready from {model_path}")

    def classify_admet(self, z: torch.Tensor) -> dict:
        """
        Classify ADMET properties for a latent vector z.
        
        Args:
            z (torch.Tensor): Latent vector [1, 128] or [batch, 128]
        Returns:
            dict: Task names with predicted probabilities.
        """
        # Ensure z has a batch dimension
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        z = z.to(self.device)

        with torch.no_grad():
            # Inside your prediction function:
            # print(f"DEBUG Z-Vector Stats: Mean={z.mean().item():.3f}, Min={z.min().item():.3f}, Max={z.max().item():.3f}")

            # Then call the model
            logits = self.model(z) # Shape: [1, 11]
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            temperature = 2.0  # T > 1 softens the distribution
            probs = torch.sigmoid(logits / temperature).cpu().squeeze(0)
        
        # print("Printing", {task: probs[i].item() for i, task in enumerate(self.task_names)})
        # Map back to task names
        return {task: probs[i].item() for i, task in enumerate(self.task_names)}
    
    def get_task_probability(self, z, task_name):
        """
        Returns the differentiable probability tensor for a specific task.
        Used primarily for Gradient Ascent.
        """
        # Ensure z is on the correct device
        z = z.to(self.device)
        
        # Forward pass (Keep the graph alive!)
        logits = self.model(z)
        
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in ADMET model.")
            
        task_idx = self.task_names.index(task_name)
        
        # Select the specific head output
        target_logit = logits[:, task_idx]
        
        # Return probability (Sigmoid)
        return torch.sigmoid(target_logit)