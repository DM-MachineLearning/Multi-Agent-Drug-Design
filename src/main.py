import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agents.HunterAgent import HunterAgent
from Agents.MedicAgent import MedicAgent

import torch

from Generators.VAE import VAE

from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine
from utils.utils import load_property_config, extract_hard_filter_keys, extract_soft_filter_keys, update_vae_backbone, get_random_scaffold_anchor

PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")
PATH_CONFIG = load_property_config("configs/paths.yaml")

NUM_GENERATIONS = 100
STEPS_PER_GENERATION = 1000

GOLDEN_SCAFFOLD = "O=C(Cc1nc(N2CCOCC2)cc(=O)[nH]1)N1CCc2ccccc21"

def main():
    vae = VAE(model_path=PATH_CONFIG['vae_model_path']) # TODO: Update model path. Take from config.
    vae.load_model(vocab_base=PATH_CONFIG['vocab_path'])

    # Initialize Scoring Engine and Blackboard
    scoring_engine = ScoringEngine(
        activity_classifier_path=PATH_CONFIG['activity_classifier_model_path'],
        admet_model_path=PATH_CONFIG['admet_model_path']
    )
    blackboard = Blackboard(config=PROPERTY_CONFIG)
    
    agents = []
    agents.append(HunterAgent(f"Potency", vae, scoring_engine, blackboard))

    # Initialize Hunter and Medic Agents based on property configuration
    hard_filters = extract_hard_filter_keys(PROPERTY_CONFIG)
    soft_filters = extract_soft_filter_keys(PROPERTY_CONFIG)
    for i in range(len(hard_filters)):
        for cnt in range(1):
            agents.append(HunterAgent(f"{hard_filters[i]}_{cnt}", vae, scoring_engine, blackboard))
    print(f"Total Hunter Agents: {len(agents)}")

    admet_properties = hard_filters + soft_filters
    for i in range(len(admet_properties)):
        agents.append(MedicAgent(f"{admet_properties[i]}", vae, scoring_engine, blackboard))
    print(f"Total Medic Agents: {len(admet_properties)}")

    print(f"Starting Discovery Pipeline: {len(agents)} agents initialized.")
    print("----------------------------------------------------------------------")

    # Main Discovery Loop
    for gen in range(NUM_GENERATIONS):
        print(f"\n--- STARTING GENERATION {gen} ---")

        for step in range(STEPS_PER_GENERATION):
            # z_anchor = get_random_scaffold_anchor(vae, PATH_CONFIG['scaffolds_path'])
            z_anchor = None
        
            if z_anchor is not None:
                blackboard.z_anchor = z_anchor
            else:
                # Fallback if file missing or encoding fails
                print("⚠️ No anchor set. Agents will search randomly.")
                blackboard.z_anchor = None

            for agent in agents:
                agent.run_step()
        
        num_hits = len(blackboard.hall_of_fame)
        tasks_left = sum(len(q) for q in blackboard.task_queues.values())
        print(f"Generation {gen} Stats:")
        print(f" - Successful Candidates (Hall of Fame): {num_hits}")
        print(f" - Unsolved Tasks on Blackboard: {tasks_left}")

        # Fine-tune VAE Backbone if enough hits are found. Write successes to CSV.
        if num_hits >= 50:
            print("The Council is meeting: Fine-tuning VAE Backbone on successes...")
            successful_zs = [item[0] for item in blackboard.hall_of_fame]
            # write_successful_molecules_to_csv(blackboard.hall_of_fame, PATH_CONFIG['successful_molecules_path'])
            update_vae_backbone(vae, successful_zs)
            blackboard.hall_of_fame = [] 

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Total Unique Drug Candidates Found: {len(blackboard.hall_of_fame)}")

    print("\n--- QUEUE DIAGNOSTICS ---")
    for tag, queue in blackboard.task_queues.items():
        print(f"Task {tag}: {len(queue)} items pending")

if __name__ == "__main__":
    main()