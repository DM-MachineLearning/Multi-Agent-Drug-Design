import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BitVectToText
import pickle


def classify_activity(smiles_list):
    """
    Classifies the activity of molecules based on their SMILES representation.
    
    Args:
        smiles_list (list of str): List of SMILES strings representing molecules.
        
    Returns:
        pd.DataFrame: DataFrame containing SMILES and their predicted activity.
    """

    with open("Multi-Agent-Drug-Design/Models/ActivityClassifier/model_Morgan_Only.pkl", 'rb') as file:
        model = pickle.load(file)

    results = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )
        fps = BitVectToText(fp)
        fingerprint_matrix = np.array([list(map(int, fp)) for fp in fps])
        output = model.predict(fingerprint_matrix.reshape(1, 2048))  # Reshaping here since it has a single sample
        results.append({"SMILES": smiles, "Activity": "Active" if output > 0.5 else "Inactive"})

    return np.array(results)