import torch

from src.madm.properties.qed import qed_from_smiles
from src.madm.properties.sa_score import sa_from_smiles

class ScoringEngine:
    def __init__(self, multitask_model, admet_model_path):
        """
        Args:
            multitask_model: The loaded MultiTaskADMETModel instance.
            task_names: List of task strings (BBBP, CYP1A2_inhibition, etc.)
        """
        self.model = multitask_model
        self.model.eval()
        self.admet_model_path = admet_model_path

    def _get_deterministic_scores(self, z):
        """Placeholders for non-ML properties."""
        return {
            "SA_score": sa_from_smiles(z),
            "QED": qed_from_smiles(z),
            "pIC50_classifier": 0.0  # TODO: Implement Potency model
        }
    
    def _get_admet_score(self, z):
        """Get ADMET scores from the multitask model."""
        with torch.no_grad():
            admet_outputs = self.model(z)
        
        admet_scores = {}
        for i, task_name in enumerate(self.model.task_names):
            admet_scores[task_name] = admet_outputs[:, i].cpu().numpy()
        
        return admet_scores

    def get_all_scores(self, z):
        """
        Returns a dict of scores for all properties in the config.
        'z' is the input tensor of shape (batch_size, 2055).
        """
        all_results = {}
        
        # 1. Get Placeholders
        all_results.update(self._get_deterministic_scores(z))
        all_results.update(self._get_admet_score(z))

        return all_results

# Example Usage:
# engine = ScoringEngine(loaded_model, TASK_NAMES)
# scores = engine.get_all_scores(test_input)