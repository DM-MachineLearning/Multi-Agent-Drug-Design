import torch

from qed import qed_from_smiles
from sa_score import sa_from_smiles
from activity_classifier import classify_activity

class ScoringEngine:
    def __init__(self, activity_classifier, admet_model_path):
        """
        Args:
            activity_classifier: The loaded ActivityClassifier instance.
            admet_model_path: Path to the ADMET model.
        """
        self.activity_classifier = activity_classifier
        self.admet_model_path = admet_model_path

    def _get_deterministic_scores(self, z):
        """Placeholders for non-ML properties."""
        return {
            "SA_score": sa_from_smiles(z),
            "QED": qed_from_smiles(z),
            "pIC50_classifier": classify_activity(z)
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