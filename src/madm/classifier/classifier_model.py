class DecisionMakingAgent:
    """
    Corresponds to the 'Decision making agent' circle in the diagram.
    It takes raw scores and performs 'Classification' into Active/Inactive.
    """
    def __init__(self, qed_weight=0.5, docking_weight=0.5, activity_threshold=0.4):
        self.w_qed = qed_weight
        self.w_dock = docking_weight
        self.threshold = activity_threshold

    def classify(self, candidate: dict) -> str:
        """
        Classifies a molecule as 'Active' or 'Inactive' based on the composite score.
        """
        # 1. Calculate Composite Score
        # Normalize Docking if needed (assuming higher is better for this formula)
        score = (candidate['QED'] * self.w_qed) + (candidate['Docking'] * self.w_dock)
        
        # 2. Apply Cross-Communication Penalty (The 'Feedback' loss component)
        # If it's a copycat (high cross-similarity), we penalize it to 0.
        if candidate.get('cross_similarity', 0) > 0.7:
            score = 0.0

        candidate['Final_Score'] = score

        # 3. Classification Step
        if score > self.threshold:
            return "Active"
        else:
            return "Inactive"