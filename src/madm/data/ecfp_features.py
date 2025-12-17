# src/madm/data/ecfp_features.py

import numpy as np
from typing import List
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator


# Fingerprint configuration

FP_RADIUS = 2
FP_BITS = 2048
N_DESCRIPTORS = 7
FEATURE_DIM = FP_BITS + N_DESCRIPTORS


# Create Morgan fingerprint generator 

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_BITS
)


# Feature extraction


def smiles_to_features(smiles: str) -> np.ndarray:
    """
    Convert a SMILES string into:
    [ECFP (2048 bits) | RDKit global descriptors (7 floats)]

    Invalid SMILES → zero vector (safe for batching).
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # --- ECFP ---
    fp = _morgan_gen.GetFingerprint(mol)
    fp_arr = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    # --- Global descriptors ---
    desc = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
        ],
        dtype=np.float32,
    )

    return np.concatenate([fp_arr.astype(np.float32), desc])


def featurize_smiles_list(smiles_list: List[str]) -> np.ndarray:
    """
    Vectorize a list of SMILES strings.

    Returns:
        np.ndarray of shape (N, 2055)
    """
    features = np.zeros((len(smiles_list), FEATURE_DIM), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        features[i] = smiles_to_features(smi)
    return features
