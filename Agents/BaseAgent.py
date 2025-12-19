PROPERTY_CONFIG = {
    'activity': {'target': 'high', 'threshold': 0.8},
    'solubility': {'target': 'high', 'threshold': -3.0}, # LogP > -3
    'toxicity': {'target': 'low', 'threshold': 0.2},
    'herg': {'target': 'low', 'threshold': 0.3},
    # ... add all 11 here
}

class BaseAgent:
    def __init__(self, agent_id, vae_backbone, scoring_engine, blackboard):
        self.id = agent_id
        self.vae = vae_backbone
        self.scorer = scoring_engine
        self.board = blackboard
        
    def gradient_ascent(self, z, objective_prop, steps=20, lr=0.1, constraint_z=None, lambda_penalty=10.0):
        """
        The Universal Optimization Function.
        - If constraint_z is None: Pure exploration (Hunter).
        - If constraint_z is set: Constrained optimization (Medic).
        """
        z = z.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        
        target_cfg = PROPERTY_CONFIG[objective_prop]
        maximize = (target_cfg['target'] == 'high')

        for _ in range(steps):
            optimizer.zero_grad()
            score = self.scorer.get_score(z, objective_prop)
            
            # 1. Base Loss (Maximize or Minimize property)
            loss = -score if maximize else score
            
            # 2. Constraint Penalty (Keep structure similar)
            if constraint_z is not None:
                dist = F.mse_loss(z, constraint_z)
                loss += lambda_penalty * dist
            
            loss.backward()
            optimizer.step()
            
        return z.detach()

    def analyze_and_route(self, z):
        """Decides if a molecule is a Success, a Failure, or a 'Fixable' task."""
        scores = self.scorer.get_all_scores(z)
        
        # Check primary success (e.g., is it active?)
        if scores['activity'] < PROPERTY_CONFIG['activity']['threshold']:
            return # Discard, not active enough to care about
            
        # If active, check for flaws (ADMET issues)
        flaws = []
        for prop, cfg in PROPERTY_CONFIG.items():
            if prop == 'activity': continue
            
            is_bad = (scores[prop] > cfg['threshold']) if cfg['target'] == 'low' else (scores[prop] < cfg['threshold'])
            if is_bad:
                flaws.append(prop)
        
        if not flaws:
            self.board.hall_of_fame.append((z, scores))
            print(f"Agent {self.id}: FOUND PERFECT MOLECULE!")
        else:
            # Post the first major flaw to the blackboard
            primary_flaw = flaws[0] # Priorities can be set here
            self.board.post_task(primary_flaw, z, scores)
            print(f"Agent {self.id}: Active but bad {primary_flaw}. Posted to Board.")