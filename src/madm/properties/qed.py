from rdkit import Chem
from rdkit.Chem import QED

def qed_from_smiles(smiles: str) -> float:
    """Compute QED (0–1, higher is better) from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(QED.qed(mol))
