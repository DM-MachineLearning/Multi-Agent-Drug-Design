"""
Molecule Generation Utility

This module provides functionality to generate molecular SMILES strings using a trained Variational Autoencoder (VAE).
It loads a pre-trained VAE model from a checkpoint, generates SMILES sequences, and validates them using RDKit
to ensure chemical validity.

The generation process involves:
- Loading the VAE model and tokenizer from a checkpoint directory
- Sampling from the latent space to generate SMILES strings
- Validating generated molecules using RDKit
- Returning a list of chemically valid SMILES strings

Usage:
    python generate.py  # Uses default checkpoint and parameters
    or
    from utils.generate import generate
    valid_smiles = generate("./trained_vae", num_molecules=50, temperature=0.7)
"""

import torch
from pathlib import Path
from Generators.VAE import VAE
from rdkit import Chem
from rdkit import RDLogger

# Disable RDKit warnings for cleaner output
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# Default checkpoint path
DEFAULT_CHECKPOINT = "./trained_vae"

def generate(checkpoint_path: str = DEFAULT_CHECKPOINT, num_molecules: int = 20, temperature: float = 0.8) -> list:
    """
    Generate molecular SMILES strings using a trained VAE model.

    Loads the VAE model from the specified checkpoint, generates SMILES sequences by sampling
    from the latent space, and validates each generated molecule using RDKit to ensure chemical validity.

    Args:
        checkpoint_path (str): Path to the checkpoint directory containing 'vae_weights.pt' and tokenizer files.
                               Defaults to './trained_vae'.
        num_molecules (int, optional): Number of SMILES strings to generate. Defaults to 20.
        temperature (float, optional): Sampling temperature for generation (lower = more deterministic).
                                       Defaults to 0.8.

    Returns:
        list: List of chemically valid SMILES strings.

    Raises:
        FileNotFoundError: If the checkpoint directory or required files are not found.
        RuntimeError: If model loading or generation fails.

    Example:
        >>> valid_smiles = generate("./trained_vae", num_molecules=10, temperature=0.6)
        >>> print(f"Generated {len(valid_smiles)} valid molecules")
    """
    print(f"Starting molecule generation with temperature={temperature}...")

    # Verify checkpoint path exists
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    model_weights = checkpoint_dir / "vae_weights.pt"
    if not model_weights.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights}")

    # Initialize the VAE manager with the model weights path
    vae_manager = VAE(model_path=str(model_weights))

    # Load the model and tokenizer from the checkpoint directory
    try:
        vae_manager.load_model(vocab_base=str(checkpoint_dir))
        vae_manager.model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")

    print(f"--- Generating {num_molecules} molecules ---")

    # Generate SMILES by sampling from the latent space N(0,1)
    try:
        generated_list = vae_manager.generate_molecule(
            num_samples=num_molecules,
            max_length=128,
            temperature=temperature
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate molecules: {e}")

    # Validate generated molecules with RDKit
    valid_molecules = []
    for smi in generated_list:
        mol = Chem.MolFromSmiles(smi)
        status = "✅ [VALID]" if mol else "❌ [INVALID]"
        print(f"{status} {smi}")
        if mol:
            valid_molecules.append(smi)

    print(f"\nSummary: {len(valid_molecules)}/{num_molecules} were chemically valid.")
    return valid_molecules

if __name__ == "__main__":
    # Generate molecules using the default checkpoint
    generate(DEFAULT_CHECKPOINT, num_molecules=20, temperature=0.6)
