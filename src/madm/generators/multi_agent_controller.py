import torch
import torch.nn.functional as F

class MultiAgentGeneratorController:
    """
    Wrap several generators and compute cross-communication loss
    to encourage diversity between them.
    """

    def __init__(self, generators):
        self.generators = list(generators)

    def sample_all(self, batch_size, condition=None):
        """Return list of (smiles, log_probs, latent) per generator."""
        outputs = []
        for gen in self.generators:
            outputs.append(gen.sample(batch_size, condition, return_latent=True))
        return outputs

    def cross_communication_loss(self, latent_list, tau: float = 0.8):
        """
        latent_list: list of tensors [batch, d], one per generator.

        Penalise cosine similarity above threshold tau.
        """
        if len(latent_list) < 2:
            return torch.tensor(0.0)

        loss = 0.0
        count = 0
        for i in range(len(latent_list)):
            for j in range(i + 1, len(latent_list)):
                e_i = F.normalize(latent_list[i], dim=-1)
                e_j = F.normalize(latent_list[j], dim=-1)
                sim = (e_i * e_j).sum(dim=-1).mean()
                loss = loss + torch.relu(sim - tau)
                count += 1
        if count > 0:
            loss = loss / count
        return loss
