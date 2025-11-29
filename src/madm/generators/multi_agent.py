import os
import torch
import numpy as np
import tempfile
import random
from pathlib import Path
from typing import List, Tuple, Set

from rdkit import Chem
from rdkit.Chem import AllChem, QED, DataStructs

# Import your existing agents
from Generators.GPT import LLM
from Generators.VAE import VAE
from madm.generators.property_agent import PropertyAgent

# ==========================================
# 1. Experience Replay Buffer (Memory)
# ==========================================
class ExperienceBuffer:
    """
    Stores high-quality molecules found across all rounds.
    Prevents 'catastrophic forgetting' by ensuring we train on a mix
    of old and new discoveries.
    """
    def __init__(self, max_size=1000):
        self.memory = set() # Use set to avoid duplicates
        self.max_size = max_size

    def add(self, smiles_list: List[str]):
        for sm in smiles_list:
            self.memory.add(sm)
            
        # Prune if too big (randomly remove to keep fresh)
        while len(self.memory) > self.max_size:
            self.memory.pop()

    def sample(self, sample_size=32) -> List[str]:
        """Returns a list of SMILES for training"""
        if len(self.memory) < sample_size:
            return list(self.memory)
        return random.sample(list(self.memory), sample_size)

    def save_buffer(self, path: Path):
        with open(path / "memory_buffer.smi", "w") as f:
            for sm in self.memory:
                f.write(sm + "\n")

# ==========================================
# 2. The Main Pipeline Controller
# ==========================================
class MultiAgentPipeline:
    def __init__(self):
        # Initialize Agents
        print("--- Initializing Agents ---")
        self.gpt = LLM(model_path=None) 
        self.gpt.load_model(use_lora=True)
        
        self.vae = VAE(model_path=None)
        self.vae.load_model()
        
        self.validator = PropertyAgent()
        self.memory = ExperienceBuffer(max_size=500)
        
        # Configuration
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.similarity_threshold = 0.7

    def _get_fp(self, smiles):
        """Helper to get fingerprint quickly"""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return None

    def generate_diverse_batch(self, generator, batch_size, opponent_fps: List, max_retries=3):
        """
        Generates a batch of molecules. If a molecule is too similar to the 'opponent'
        (the other generator), it rejects it and retries.
        """
        accepted_molecules = []
        accepted_fps = []
        
        attempts = 0
        while len(accepted_molecules) < batch_size and attempts < max_retries:
            # How many do we still need?
            needed = batch_size - len(accepted_molecules)
            
            # Generate a slightly larger batch to account for failures
            candidates = [generator.generate_molecule(temperature=0.8 + (attempts*0.1)) for _ in range(needed + 2)]
            
            for sm in candidates:
                if len(accepted_molecules) >= batch_size:
                    break
                    
                fp = self._get_fp(sm)
                if fp is None: continue # Invalid molecule
                
                # CHECK 1: Is it unique within its own batch?
                # (Optional, but good practice)
                
                # CHECK 2: Cross-Communication (Is it too similar to Opponent?)
                is_too_similar = False
                if opponent_fps:
                    # Calculate similarity against ALL opponent fingerprints
                    sims = DataStructs.BulkTanimotoSimilarity(fp, opponent_fps)
                    if max(sims) > self.similarity_threshold:
                        is_too_similar = True
                
                if not is_too_similar:
                    accepted_molecules.append(sm)
                    accepted_fps.append(fp)
            
            attempts += 1
            if len(accepted_molecules) < batch_size:
                print(f"   > Collision detected. Regenerating {batch_size - len(accepted_molecules)} molecules...")

        return accepted_molecules

    def run_optimization_loop(self, rounds=5, batch_size=50):
        for r in range(rounds):
            print(f"\n=== Round {r+1}/{rounds} ===")
            
            # --- Step 1: Sequential Generation with Collision Check ---
            # We let GPT go first (arbitrary choice), then VAE must respect GPT's space
            print(">> Generators producing molecules...")
            
            # 1. GPT generates freely
            raw_gpt = [self.gpt.generate_molecule(temperature=0.8) for _ in range(batch_size)]
            # Calculate GPT fingerprints to pass to VAE
            gpt_fps = [self._get_fp(sm) for sm in raw_gpt if self._get_fp(sm) is not None]
            
            # 2. VAE generates, but MUST NOT match GPT (Regeneration Loop)
            print(">> VAE generating (avoiding GPT collisions)...")
            raw_vae = self.generate_diverse_batch(self.vae, batch_size, opponent_fps=gpt_fps)

            # 3. (Optional) GPT Double Check 
            # If you want strict fairness, you could re-check GPT against VAE here, 
            # but usually one-way check is sufficient for diversity.

            # --- Step 2: Validation ---
            print(">> Validating scores...")
            data_gpt = self.validator.evaluate_batch(raw_gpt)
            data_vae = self.validator.evaluate_batch(raw_vae)
            
            # --- Step 3: Decision Making ---
            # Now we don't need to penalize similarity heavily here because 
            # we already filtered the worst offenders in Step 1.
            
            all_candidates = data_gpt + data_vae
            round_winners = []

            for item in all_candidates:
                # Pure quality score now
                final_score = (item['QED'] * 0.5) + (item['Docking'] * 0.5)
                
                if final_score > 0.4: 
                    round_winners.append(item['smiles'])

            # --- Step 4: Feedback (Memory + Training) ---
            print(f">> Feedback: Found {len(round_winners)} active candidates.")
            
            # A. Update Memory
            self.memory.add(round_winners)
            
            # B. Train on Experience Buffer (Not just new dataset)
            # This answers "How do I ensure it generates more such molecules?"
            if len(self.memory.memory) > 10:
                print(f">> Improving Generators using Experience Replay (Size: {len(self.memory.memory)})...")
                
                # Sample a mix of old and new good molecules
                training_samples = self.memory.sample(sample_size=32)
                
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi') as tmp:
                    for sm in training_samples:
                        tmp.write(sm + "\n")
                    tmp_path = tmp.name

                try:
                    # Low learning rate to gently nudge weights
                    self.gpt.fine_tune(tmp_path, epochs=1, batch_size=4, lr=1e-5)
                    self.vae.fine_tune(tmp_path, epochs=1, batch_size=8, lr=1e-4)
                finally:
                    os.remove(tmp_path)
            else:
                print(">> Memory buffer too small to train yet.")

    def save_final_models(self):
        print("Saving final models and memory...")
        self.gpt.model.save_pretrained(self.output_dir / "Final_GPT")
        torch.save(self.vae.model.state_dict(), self.output_dir / "Final_VAE.pt")
        self.memory.save_buffer(self.output_dir)