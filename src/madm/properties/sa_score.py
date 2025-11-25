from rdkit import Chem

# NOTE:
# Place the Ertl SA score implementation from RDKit contrib as _sascorer.py
# in this same directory. It should expose a function `calculateScore(mol)`.
# Here we just wrap it.
try:
    from . import _sascorer
except ImportError:
    _sascorer = None

def sa_from_smiles(smiles: str) -> float:
    """
    Compute synthetic accessibility (SA) score.

    Lower scores ~1–3 are easy to synthesize.
    Higher scores ~7–10 are challenging.

    If the underlying scorer is missing or SMILES is invalid,
    we conservatively return a high (unfavourable) score.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or _sascorer is None:
        return 10.0
    return float(_sascorer.calculateScore(mol))
