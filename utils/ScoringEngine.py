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

    # def get_all_scores(self, z):
    #     """
    #     Get all relevant scores for a given molecule.
    #     """
    #     return {
    #         # "SA_score": sa_from_smiles(z),
    #         # "QED": qed_from_smiles(z),
    #         "pIC50_classifier": self.activity_classifier_model.classify_activity(z),
    #         "admet_scores": self.admet_classifier_model.classify_admet(z)
    #     }

    def get_all_scores(self, z):
        """
        CRITICAL: This must accept 'z' and pass it down.
        """
        scores = {}
        
        # 1. Get ADMET scores (using the new tensor-based method)
        # This returns a dictionary of Tensors { 'BBBP': tensor(0.95, grad_fn=...), ... }
        scores.update(self.admet_classifier_model.classify_admet(z))
        
        # 2. Get Potency score (Ensure this also accepts z if possible, or decode)
        # If your potency model still needs SMILES, you have a bottleneck here.
        # Ideally, convert potency model to latent-based too.
        scores['potency'] = self.activity_classifier_model.classify_activity(z) 
        
        return scores