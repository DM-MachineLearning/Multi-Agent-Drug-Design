import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BitVectToText

from Models.ActivityClassifier.train_mlp import LatentPredictor

class ActivityClassifier:
    """
    A classifier for predicting the activity of molecules based on their SMILES representation.
    The model is a pre-trained machine learning model loaded from a specified path. The model
    is based on random forest algorithm. It uses Morgan fingerprints as features for classification.

    Attributes:
        model: The pre-trained machine learning model for activity classification.
    """
    def __init__(self, model_path = "Models/ActivityClassifier/checkpoints/activity_classifier_mlp.pt"):
        """
        Initializes the ActivityClassifier by loading a pre-trained model.
        Args:
            model_path (str): The file path to the pre-trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LatentPredictor(input_dim=128).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def classify_activity(self, z):
        """
        Classifies the activity of a molecule based on its SMILES representation.

        Args:
            smiles (str): The SMILES representation of the molecule.

        Returns:
            dict: A dictionary containing the predicted activity.
        """
        # mol = Chem.MolFromSmiles(smiles) # Convert SMILES to RDKit molecule
        # if mol is None:
        #     raise ValueError("Invalid SMILES string provided.")

        # # Generate Morgan fingerprint
        # fp = AllChem.GetMorganFingerprintAsBitVect(
        #     mol,
        #     radius=2,
        #     nBits=2048
        # )

        # # Convert fingerprint to text representation
        # fps = BitVectToText(fp)
        
        # # Convert fingerprint text to numpy array
        # fingerprint_matrix = np.array([list(map(int, fp)) for fp in fps])
        
        # Predict activity using the pre-trained model
        # output = self.model(torch.tensor(smiles, dtype=torch.float32).to(self.device))
        
        # return {"SMILES": smiles, "Activity": "Active" if output > 0.5 else "Inactive"}

        logits = self.model(z)
        return torch.sigmoid(logits)