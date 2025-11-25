from rdkit import Chem
from rdkit.Chem import AllChem
import torch

def smiles_to_mol(smiles: str):
    """Convert a SMILES string to an RDKit Mol object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol

def mol_to_ecfp(mol, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    """Return ECFP (Morgan) fingerprint as a float tensor of 0/1."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = torch.tensor(list(fp), dtype=torch.float32)
    return arr

def smiles_to_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    mol = smiles_to_mol(smiles)
    return mol_to_ecfp(mol, radius=radius, n_bits=n_bits)
