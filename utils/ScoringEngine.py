class ScoringEngine:
    def __init__(self, models_dict):
        self.models = models_dict # {'activity': model_A, 'solubility': model_B, ...}

    def get_score(self, z, property_name):
        """Returns the score for a specific property."""
        # Assuming models output a single scalar
        return self.models[property_name](z)

    def get_all_scores(self, z):
        """Returns a dict of scores for all 11 properties."""
        scores = {}
        for name, model in self.models.items():
            scores[name] = model(z).item()
        return scores