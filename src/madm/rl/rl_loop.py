"""Skeleton RL training loop for the multi-agent generator block."""
from typing import List, Tuple
import torch
from ..data.featurization import smiles_to_ecfp
from ..properties.property_agent import PropertyValidatingAgent
from .reward_functions import total_reward

def rl_training_step(controller,
                     property_agent: PropertyValidatingAgent,
                     classifier,
                     optimizer,
                     batch_size: int,
                     cross_comm_weight: float = 0.1) -> float:
    """
    Perform one RL update step over all generators.

    This is a high-level sketch; you will need to adapt it to your
    generator implementation and data structures.
    """
    controller_outputs = controller.sample_all(batch_size)

    all_rewards = []
    all_log_probs = []
    latent_list = []

    for smiles_list, log_probs, latents in controller_outputs:
        # Compute properties for each molecule
        fps = []
        prop_vecs = []
        props_dict = {"qed": [], "sa": [], "dock": [], "admet": []}

        for smi in smiles_list:
            props = property_agent.compute(smi)
            fp = smiles_to_ecfp(smi)
            fps.append(fp)
            prop_vecs.append(
                torch.tensor(
                    [props["qed"], props["sa"], props["admet"], props["dock"]],
                    dtype=torch.float32,
                )
            )
            for k in props_dict:
                props_dict[k].append(props[k])

        fp_batch = torch.stack(fps, dim=0)
        prop_batch = torch.stack(prop_vecs, dim=0)
        props_tensor_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in props_dict.items()}

        reward = total_reward(classifier, fp_batch, prop_batch, props_tensor_dict)
        all_rewards.append(reward)
        all_log_probs.append(log_probs)
        latent_list.append(latents)

    cross_loss = controller.cross_communication_loss(latent_list)
    total_loss = 0.0

    for log_probs, reward in zip(all_log_probs, all_rewards):
        gen_loss = -(log_probs * reward.detach()).mean()
        total_loss = total_loss + gen_loss

    total_loss = total_loss + cross_comm_weight * cross_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return float(total_loss.item())
