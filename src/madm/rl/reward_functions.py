import torch

def reward_from_classifier(classifier, fp_batch, prop_batch):
    """Reward = P(active) from classifier (0–1)."""
    with torch.no_grad():
        p_active = classifier.active_prob(fp_batch, prop_batch)
    return p_active

def property_penalty(props, qed_min: float = 0.4, sa_max: float = 6.0):
    """
    Soft penalties:

    - Penalise QED below qed_min.
    - Penalise SA above sa_max.
    """
    qed = props["qed"]
    sa = props["sa"]
    qed_term = torch.relu(qed_min - qed)
    sa_term = torch.relu(sa - sa_max)
    return -(qed_term + sa_term)

def total_reward(classifier, fp_batch, prop_batch, props_dict):
    base = reward_from_classifier(classifier, fp_batch, prop_batch)
    pen = property_penalty(props_dict)
    return base + pen
