import yaml
import torch
import torch.nn.functional as F
import csv
import os
import random

# def write_successful_molecules_to_csv(hall_of_fame, filepath):
#     """
#     Writes successful molecules from the hall of fame to a CSV file.
    
#     Args:
#         hall_of_fame: List of tuples containing (z_vector, molecule_smiles, score).
#         filepath: Path to the CSV file where data will be appended.
#     """
#     with open(filepath, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for item in hall_of_fame:
#             writer.writerow([item[1], item[2]])

def write_successful_molecules_to_csv(hall_of_fame, filepath, vae=None):
    """
    Writes successful leads to CSV. 
    hall_of_fame: list of tuples -> (z, scores)
    """
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writerow(['smiles', 'captions']) 
            
        for item in hall_of_fame:
            # --- FIX 1: Handle the tuple correctly (Only 2 items) ---
            z = item[0]       # The Latent Vector
            scores = item[1]  # The Scores Dictionary
            
            # --- FIX 2: Translate z to SMILES ---
            # We use generate_molecule(z=z) which acts as a "Decoder" 
            # when you provide the z. It does NOT make a new random molecule.
            if vae is not None:
                # generate_molecule returns (smiles, z). We only need the smiles.
                smi, _ = vae.generate_molecule(z=z)
            else:
                smi = "Latent_Vector_Only"
            
            # Save to CSV
            writer.writerow([smi, str(scores)])

    print(f"✅ Successfully wrote {len(hall_of_fame)} leads to {filepath}")

# def update_vae_backbone(vae, training_samples, kl_weight=0.01):
#     """
#     Fine-tunes the VAE weights based on successful latent points (z_target).
    
#     Args:
#         vae: The VAE model instance.
#         training_samples: A list or Tensor of latent vectors (z) that performed well.
#         kl_weight: Coefficient to prevent the latent space from collapsing.
#     """
#     if not training_samples:
#         print("No training samples provided. Skipping VAE update.")
#         return

#     vae.train()
#     optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)
    
#     if isinstance(training_samples, list):
#         z_targets = torch.stack(training_samples)
#     else:
#         z_targets = training_samples

#     print(f"Fine-tuning VAE on {len(z_targets)} successful samples...")

#     for _ in range(2):
#         optimizer.zero_grad()

#         mu, logvar = vae.encode_z(z_targets)
#         z_reparam = vae.reparameterize(mu, logvar)

#         recon_loss = F.mse_loss(z_reparam, z_targets)
        
#         kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         kl_loss = kl_loss / z_targets.size(0)
        
#         total_loss = recon_loss + (kl_weight * kl_loss)
        
#         total_loss.backward()
#         optimizer.step()
    
#     vae.eval()
#     print("VAE weights updated and synchronized across all agents.")

def update_vae_backbone(vae, successful_zs):
    """
    Instead of retraining weights (which risks catastrophic forgetting),
    we shift the VAE's sampling distribution to focus on the region
    where successes were found.
    """
    if not successful_zs:
        print("⚠️ No successes to update VAE. Keeping standard exploration.")
        return

    print("⚖️ The Council is analyzing the successful latent vectors...")

    # 1. Stack all vectors into a tensor
    # successful_zs is a list of tensors, likely on different devices or CPU
    # Ensure they are all on the same device before stacking
    device = vae.device
    z_stack = torch.stack([z.to(device).squeeze() for z in successful_zs])
    
    # 2. Calculate Statistics
    # We want to know where the "Gold" is (Mean) and how spread out it is (Std)
    new_mean = torch.mean(z_stack, dim=0)
    new_std = torch.std(z_stack, dim=0)
    
    # 3. Safety Clamp
    # Don't let the std get too small (mode collapse) or too large (random noise)
    new_std = torch.clamp(new_std, min=0.5, max=1.5)
    
    # 4. Update the VAE's internal compass
    # This assumes you added the 'update_search_distribution' method to your VAE class
    if hasattr(vae, "update_search_distribution"):
        vae.update_search_distribution(new_mean, new_std)
    else:
        print("❌ Error: VAE class missing 'update_search_distribution' method.")
        
    print(f"✅ VAE Backbone Updated: Sampling focus shifted to new high-yield region.")

def get_random_scaffold_anchor(vae, scaffolds_path):
    """
    Reads a file of scaffolds, picks one at random, and encodes it 
    into a latent vector (z_anchor) using the VAE's specific GRU architecture.

    Args:
        vae: The initialized VAE wrapper object.
        scaffolds_path (str): Path to the text file containing SMILES (one per line).

    Returns:
        torch.Tensor: The latent mean (mu) of the selected scaffold.
    """
    if not os.path.exists(scaffolds_path):
        print(f"⚠️ Scaffold file not found at {scaffolds_path}. Using random initialization.")
        return None

    # 1. Load and Sample
    with open(scaffolds_path, "r") as f:
        # Filter out empty lines just in case
        scaffolds = [line.strip() for line in f.readlines() if line.strip()]
    
    if not scaffolds:
        print("⚠️ Scaffold file is empty.")
        return None

    selected_scaffold = random.choice(scaffolds)
    # print(f"⚓ Selected Anchor Scaffold: {selected_scaffold}")

    # 2. Encode using the specific MolGRUVAE logic
    # We use the tokenizer from the VAE wrapper
    try:
        tokens = vae.tokenizer.encode(selected_scaffold, return_tensors="pt").to(vae.device)

        with torch.no_grad():
            # Embed
            embedded = vae.model.embedding(tokens)
            
            # Run GRU (Forward + Backward hidden states)
            _, h_n = vae.model.encoder_gru(embedded)
            
            # Concatenate the bidirectional states
            # h_n shape is [2, batch, hidden] -> [0] is FWD, [1] is BWD
            h_n_concat = torch.cat((h_n[0], h_n[1]), dim=1)
            
            # Project to Latent Space (Mean)
            mu = vae.model.fc_mu(h_n_concat)
            
        # print(f"✅ Encoded Anchor z-vector (Mean: {mu.mean().item():.4f})")
        return mu

    except Exception as e:
        print(f"❌ Failed to encode scaffold '{selected_scaffold}': {e}")
        return None
    
def load_property_config(filepath="properties.yaml"):
    """
    Loads the property configuration from a YAML file.
    
    How to use:
    PROPERTY_CONFIG = load_property_config("properties.yaml")
    """
    with open(filepath, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error loading YAML: {exc}")
            return None
    
    print(config)

def get_property_details(config, property_name):
    """
    Recursively searches for a property name in a nested configuration.
    Returns the dictionary associated with that property.

    How to use:
    property_details = get_property_details(PROPERTY_CONFIG, "solubility")
    """
    if property_name in config:
        if isinstance(config[property_name], dict) and 'target' in config[property_name]:
            return config[property_name]

    for key, value in config.items():
        if isinstance(value, dict):
            if property_name in value:
                return value[property_name]
    
    return None

def extract_property_keys(config):
    """
    Recursively finds all keys that represent actual properties 
    (leaves of the config tree that contain 'target' or 'threshold').
    """
    keys = []
    for key, value in config.items():
        if isinstance(value, dict):
            if 'target' in value or 'threshold' in value:
                keys.append(key)
            else:
                keys.extend(extract_property_keys(value))
    return keys

def extract_hard_filter_keys(config):
    """
    Extracts property names from the 'hard_filters' section.
    """
    # Simply get the keys from the 'hard_filters' dictionary if it exists
    hard_section = config.get('hard_filters', {})
    return list(hard_section.keys())

def extract_soft_filter_keys(config):
    """
    Extracts property names from the 'soft_filters' section.
    """
    # Get the keys from the 'soft_filters' dictionary
    soft_section = config.get('soft_filters', {})
    return list(soft_section.keys())

def extract_deterministic_keys(config):
    """
    Recursively finds all keys that have 'deterministic' set to True.
    """
    keys = []
    for key, value in config.items():
        if isinstance(value, dict):
            if value.get('deterministic', False):
                keys.append(key)
            else:
                keys.extend(extract_deterministic_keys(value))
    return keys