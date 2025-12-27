import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BitVectToText
import pickle

class ActivityClassifier:
    def __init__(self, model_path: str):
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
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )
        fps = BitVectToText(fp)
        fingerprint_matrix = np.array([list(map(int, fp)) for fp in fps])
        output = self.model.predict(fingerprint_matrix.reshape(1, 2048))
        return {"SMILES": smiles, "Activity": "Active" if output > 0.5 else "Inactive"}