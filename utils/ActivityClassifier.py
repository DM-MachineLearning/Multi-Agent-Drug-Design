import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BitVectToText
import pickle

class ActivityClassifier:
    """
    A classifier for predicting the activity of molecules based on their SMILES representation.
    The model is a pre-trained machine learning model loaded from a specified path. The model
    is based on random forest algorithm. It uses Morgan fingerprints as features for classification.

    Attributes:
        model: The pre-trained machine learning model for activity classification.
    """
    def __init__(self, model_path: str):
        """
        Initializes the ActivityClassifier by loading a pre-trained model.
        Args:
            model_path (str): The file path to the pre-trained model.
        """
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def classify_activity(self, smiles: str) -> dict:
        """
        Classifies the activity of a molecule based on its SMILES representation.

        Args:
            smiles (str): The SMILES representation of the molecule.

        Returns:
            dict: A dictionary containing the predicted activity.
        """
        mol = Chem.MolFromSmiles(smiles) # Convert SMILES to RDKit molecule
        if mol is None:
            raise ValueError("Invalid SMILES string provided.")

        # Generate Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )

        # Convert fingerprint to text representation
        fps = BitVectToText(fp)
        
        # Convert fingerprint text to numpy array
        fingerprint_matrix = np.array([list(map(int, fp)) for fp in fps])
        
        # Predict activity using the pre-trained model
        output = self.model.predict(fingerprint_matrix.reshape(1, 2048))
        
        return {"SMILES": smiles, "Activity": "Active" if output > 0.5 else "Inactive"}