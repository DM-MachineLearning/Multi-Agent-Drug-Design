from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

import math
import pickle
import gzip
import os.path as op

# The file `fpscores.pkl.gz` (RDKit/Novartis fragment scores) should
# be placed alongside this module. If missing, the function will
# conservatively return 10.0 for invalid/missing resources.

_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def _readFragmentScores(name: str = "fpscores.pkl.gz"):
    """Load fragment score dictionary from a gzipped pickle.

    Expected format: list of tuples where first element is score
    and remaining are fragment ids, matching the original implementation.
    """
    global _fscores
    if name == "fpscores.pkl.gz":
        name = op.join(op.dirname(__file__), name)
    try:
        data = pickle.load(gzip.open(name))
    except Exception:
        _fscores = None
        return
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def _numBridgeheadsAndSpiro(mol):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def _calculateScore(m):
    """Calculate raw SAS score for an RDKit Mol object.

    Returns None if molecule has no atoms or if fragment scores are missing.
    """
    if not m or not m.GetNumAtoms():
        return None

    global _fscores
    if _fscores is None:
        _readFragmentScores()
    if _fscores is None:
        # missing fragment scores file
        return None

    # fragment score using Morgan fingerprint counts
    sfp = mfpgen.GetSparseCountFingerprint(m)

    score1 = 0.0
    nf = 0
    nze = sfp.GetNonzeroElements()
    for id, count in nze.items():
        nf += count
        score1 += _fscores.get(id, -4) * count
    if nf == 0:
        return None
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = _numBridgeheadsAndSpiro(m)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for fingerprint density (symmetry)
    score3 = 0.0
    numBits = len(nze)
    if numBits > 0 and nAtoms > numBits:
        score3 = math.log(float(nAtoms) / numBits) * 0.5

    sascore = score1 + score2 + score3

    # transform raw score to 1..10 scale
    min_raw = -4.0
    max_raw = 2.5
    sascore = 11.0 - (sascore - min_raw + 1.0) / (max_raw - min_raw) * 9.0

    # smooth the high end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore


def sa_from_smiles(smiles: str) -> float:
    """Compute Synthetic Accessibility (SA) score for a SMILES string.

    Returns a float in the range ~1 (easy) to ~10 (hard). If the SMILES
    is invalid or fragment scores are not available, returns `10.0`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    s = _calculateScore(mol)
    if s is None:
        return 10.0
    return float(s)


__all__ = ["sa_from_smiles"]
