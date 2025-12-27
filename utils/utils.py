import yaml
import torch
import torch.nn.functional as F

def update_vae_backbone(vae, training_samples, kl_weight=0.01):
    """
    Fine-tunes the VAE weights based on successful latent points (z_target).
    
    Args:
        vae: The VAE model instance.
        training_samples: A list or Tensor of latent vectors (z) that performed well.
        kl_weight: Coefficient to prevent the latent space from collapsing.
    """
    if not training_samples:
        print("No training samples provided. Skipping VAE update.")
        return

    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)
    
    if isinstance(training_samples, list):
        z_targets = torch.stack(training_samples)
    else:
        z_targets = training_samples

    print(f"Fine-tuning VAE on {len(z_targets)} successful samples...")

    for _ in range(2):
        optimizer.zero_grad()

        mu, logvar = vae.encode_z(z_targets)
        z_reparam = vae.reparameterize(mu, logvar)

        recon_loss = F.mse_loss(z_reparam, z_targets)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / z_targets.size(0)
        
        total_loss = recon_loss + (kl_weight * kl_loss)
        
        total_loss.backward()
        optimizer.step()
    
    vae.eval()
    print("VAE weights updated and synchronized across all agents.")

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
    Recursively finds all keys that have 'hard_filter' set to True.
    """
    keys = []
    for key, value in config.items():
        if isinstance(value, dict):
            if value.get('hard_filter', False):
                keys.append(key)
            else:
                keys.extend(extract_hard_filter_keys(value))
    return keys

def extract_soft_filter_keys(config):
    """
    Recursively finds all keys that have 'soft_filter' set to True.
    """
    keys = []
    for key, value in config.items():
        if isinstance(value, dict):
            if value.get('soft_filter', False):
                keys.append(key)
            else:
                keys.extend(extract_soft_filter_keys(value))
    return keys

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