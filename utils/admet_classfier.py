import torch

from src.madm.properties.multitask_mlp import MultiTaskADMETModel

class ADMETClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.task_names = [
            'BBBP', 'Caco2_permeability', 'CYP1A2_inhibition', 'CYP2C9_inhibition', 
            'CYP2C19_inhibition', 'CYP2D6_inhibition', 'CYP3A4_inhibition', 
            'HERG_inhibition', 'HLM_stability', 'P-gp_substrate', 'RLM_stability'
        ]
        self.input_dim = 2055  # Changed from 2048 to 2055 based on the error
        self.hidden_dims = [512, 256] 
    
        self.model = MultiTaskADMETModel(
            input_dim=self.input_dim,
            shared_hidden_dims=self.hidden_dims,
            task_names=self.task_names
        )

        self.state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.new_state_dict = {}
        for key, value in self.state_dict.items():
            if key.startswith("trunk."):
                new_key = key.replace("trunk.", "trunk.network.", 1)
                self.new_state_dict[new_key] = value
            else:
                self.new_state_dict[key] = value

        self.model.load_state_dict(self.new_state_dict)
        self.model.eval()
        print(f"✅ Model weights loaded successfully with INPUT_DIM={self.input_dim}.")

    def classify_admet(self, smiles: str) -> dict:
        """
        Classify ADMET properties for a given SMILES string.

        Args:
            smiles (str): The SMILES representation of the molecule.
        Returns:
            dict: A dictionary with task names as keys and predicted probabilities as values.
        """
        with torch.no_grad():
            logits_dict = self.model(smiles)

        probabilities = {}
        for task, logits in logits_dict.items():
            prob = torch.sigmoid(logits).item()
            probabilities[task] = prob

        return probabilities

# 1. Configuration - Updated to match the checkpoint shape
# TASK_NAMES = [
#     'BBBP', 'Caco2_permeability', 'CYP1A2_inhibition', 'CYP2C9_inhibition', 
#     'CYP2C19_inhibition', 'CYP2D6_inhibition', 'CYP3A4_inhibition', 
#     'HERG_inhibition', 'HLM_stability', 'P-gp_substrate', 'RLM_stability'
# ]
# INPUT_DIM = 2055  # Changed from 2048 to 2055 based on the error
# HIDDEN_DIMS = [512, 256] 
# MODEL_PATH = 'outputs_multitask/multitask_model.pt'

# # 2. Initialize the model
# model = MultiTaskADMETModel(
#     input_dim=INPUT_DIM,
#     shared_hidden_dims=HIDDEN_DIMS,
#     task_names=TASK_NAMES
# )

# 3. Load and Remap State Dict
# try:
#     state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
#     # Map the old keys to the new class structure
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         if key.startswith("trunk."):
#             new_key = key.replace("trunk.", "trunk.network.", 1)
#             new_state_dict[new_key] = value
#         else:
#             new_state_dict[key] = value

#     model.load_state_dict(new_state_dict)
#     model.eval()
#     print("✅ Model weights loaded successfully with INPUT_DIM=2055.")
# except Exception as e:
#     print(f"❌ Loading failed: {e}")
#     exit()
