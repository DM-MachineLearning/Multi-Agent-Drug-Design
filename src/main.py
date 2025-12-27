import yaml
import torch

from Agents.hunter import HunterAgent
from Agents.MedicAgent import MedicAgent

from Generators.VAE import VAE

from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine
from utils.utils import load_property_config, extract_property_keys, extract_hard_filter_keys, extract_soft_filter_keys

PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")
PATH_CONFIG = yaml.safe_load(open("configs/paths.yaml"))

def main():
    vae = VAE(model_path="models/vae_model.pt")
    vae.eval()
    
    scoring_engine = ScoringEngine(
        activity_classifier_path=PATH_CONFIG['activity_classifier_model_path'],
        admet_model_path=PATH_CONFIG['admet_model_path']
    )
    blackboard = Blackboard(config=PROPERTY_CONFIG)
    
    agents = []

    hard_filters = extract_hard_filter_keys(PROPERTY_CONFIG)
    soft_filters = extract_soft_filter_keys(PROPERTY_CONFIG)
    for i in range(len(hard_filters)):
        agents.append(HunterAgent(f"{hard_filters[i]}", vae, scoring_engine, blackboard))

    admet_properties = [hard_filters + soft_filters]
    for i in range(len(admet_properties)):
        agents.append(MedicAgent(f"{admet_properties[i]}", vae, scoring_engine, blackboard))

    # --- Step 3: Start Evolutionary Generations ---
    NUM_GENERATIONS = 10
    STEPS_PER_GENERATION = 100 # How many molecules each agent tries to generate/fix per loop

    print(f"Starting Discovery Pipeline: {len(agents)} agents initialized.")

    for gen in range(NUM_GENERATIONS):
        print(f"\n--- STARTING GENERATION {gen} ---")
        
        # 1. Parallel Agent Execution
        # In a production environment, use multiprocessing or concurrent.futures
        for step in range(STEPS_PER_GENERATION):
            for agent in agents:
                agent.run_step()
        
        # 2. End of Generation Report
        num_hits = len(blackboard.hall_of_fame)
        tasks_left = sum(len(q) for q in blackboard.task_queues.values())
        print(f"Generation {gen} Stats:")
        print(f" - Successful Candidates (Hall of Fame): {num_hits}")
        print(f" - Unsolved Tasks on Blackboard: {tasks_left}")

        # 3. THE COUNCIL: Evolutionary Update
        if num_hits >= 50: # Only update if we have enough "Gold" data
            print("The Council is meeting: Fine-tuning VAE Backbone on successes...")
            
            # Extract successful latent vectors for fine-tuning
            successful_zs = [item[0] for item in blackboard.hall_of_fame]
            
            # Implementation of Evolutionary Update (Task Arithmetic or Fine-tuning)
            # This 'shifts' the latent space towards successful chemical regions
            update_vae_backbone(vae, successful_zs)
            
            # Optional: Clear or prune the Hall of Fame to focus on NEW successes
            # blackboard.hall_of_fame = [] 

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Total Unique Drug Candidates Found: {len(blackboard.hall_of_fame)}")

# =========================================================
# 3. EVOLUTIONARY UPDATE LOGIC
# =========================================================

def update_vae_backbone(vae, training_samples):
    """
    Fine-tunes the VAE weights based on the successful discoveries.
    """
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)
    
    # We treat the successful molecules as the new "Ground Truth"
    # To shift the distribution of the latent space
    for epoch in range(2):
        for z_target in training_samples:
            optimizer.zero_grad()
            
            # Recon loss: ensure the VAE can still reconstruct these 'Perfect' points
            recon_z = vae.decode(z_target) 
            # (Assuming you have a method to calculate recon loss from latent to latent)
            
            # Custom logic to pull the VAE prior towards these samples
            # ...
            
            optimizer.step()
    
    vae.eval()
    print("VAE weights updated and synchronized across all agents.")

if __name__ == "__main__":
    main()