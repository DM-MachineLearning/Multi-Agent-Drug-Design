from utils.qed import qed_from_smiles
from utils.sa_score import sa_from_smiles
from utils.ActivityClassifier import ActivityClassifier
from utils.ADMETClassifier import ADMETClassifier

class ScoringEngine:
    def __init__(self, activity_classifier_path, admet_model_path):
        """
        Args:
            activity_classifier: The loaded ActivityClassifier instance.
            admet_model_path: Path to the ADMET model.
        """
        self.activity_classifier_model = ActivityClassifier(activity_classifier_path)
        self.admet_classifier_model = ADMETClassifier(admet_model_path)

    def get_all_scores(self, z):
        """
        Get all relevant scores for a given molecule.
        """
        return {
            "SA_score": sa_from_smiles(z),
            "QED": qed_from_smiles(z),
            "pIC50_classifier": self.activity_classifier_model.classify_activity(z),
            "admet_scores": self.admet_classifier_model.classify_admet(z)
        }