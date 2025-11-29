import os
import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, QED, DataStructs

class PropertyAgent:
    """
    Handles the 'Feedback' cloud and 'Property validating agent' block.
    Calculates QED, ADMET (Simulated), and Similarity metrics.
    """
    def __init__(self):
        pass

    def get_fingerprint(self, smiles: str):
        """Converts SMILES to Morgan Fingerprint (Bit Vector)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return None

    def calculate_tanimoto(self, fp1, fp2) -> float:
        """Calculates Similarity between two fingerprints."""
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def calculate_docking_score(self, smiles: str) -> float:
        """
        [MOCK] Interface for external Docking (e.g., AutoDock Vina).
        Replace this with your actual subprocess call.
        """
        # TODO: Implement actual docking here. 
        # For now, we simulate: Longer carbon chains ~ better score (just for testing flow)
        if not smiles: return 0.0
        return min(1.0, len(smiles) / 50.0) 

    def evaluate_batch(self, smiles_list: List[str]) -> List[dict]:
        """
        Runs the full validation pipeline (QED -> ADMET -> Docking).
        Returns a list of dictionaries with scores.
        """
        results = []
        for sm in smiles_list:
            mol = Chem.MolFromSmiles(sm)
            if mol is None:
                continue # Discard invalid syntax immediately

            # 1. QED (Drug-likeness)
            qed_score = QED.qed(mol)

            # 2. Mock Docking
            docking_score = self.calculate_docking_score(sm)

            results.append({
                "smiles": sm,
                "fingerprint": self.get_fingerprint(sm),
                "QED": qed_score,
                "Docking": docking_score,
                "Final_Score": 0.0 # Will be calculated after cross-loss
            })
        return results