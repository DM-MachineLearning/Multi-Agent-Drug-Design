import torch

from qed import qed_from_smiles
from sa_score import sa_from_smiles
from utils.activity_classifier import ActivityClassifier
from utils.admet_classfier import ADMETClassifier

class ScoringEngine:
    def __init__(self, activity_classifier_path, admet_model_path):
        """
        Args:
            activity_classifier: The loaded ActivityClassifier instance.
            admet_model_path: Path to the ADMET model.
        """
        self.activity_classifier_model = ActivityClassifier(activity_classifier_path)
        self.admet_classifier_model = ADMETClassifier(admet_model_path)

    def _get_scores(self, z):
        return {
            "SA_score": sa_from_smiles(z),
            "QED": qed_from_smiles(z),
            "pIC50_classifier": self.activity_classifier_model.classify_activity(z),
            "admet_scores": self.admet_classifier_model.classify_admet(z)
        }

    def get_all_scores(self, z):
        """
        Returns a dict of scores for all properties in the config.
        'z' is the input tensor of shape (batch_size, 2055).
        """
        all_results = {}
        
        # 1. Get Placeholders
        all_results.update(self._get_scores(z))

        return all_results