from dataclasses import dataclass
from typing import Dict, Any
import torch
from ..data.featurization import smiles_to_ecfp
from .qed import qed_from_smiles
from .sa_score import sa_from_smiles
from .admet_placeholder import admet_from_smiles

@dataclass
class PropertyValidatingAgent:
    """Compute QED, SA, ADMET and docking score for a SMILES string."""
    docking_model: Any  # expected to be DockingRegressor or similar

    def compute(self, smiles: str) -> Dict[str, float]:
        qed = qed_from_smiles(smiles)
        sa = sa_from_smiles(smiles)
        admet = admet_from_smiles(smiles)
        fp = smiles_to_ecfp(smiles)
        dock = self.docking_model.predict(fp)
        return {"qed": qed, "sa": sa, "admet": admet, "dock": dock}
