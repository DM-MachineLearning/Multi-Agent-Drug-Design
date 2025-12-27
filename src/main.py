import csv

from Agents.hunter import HunterAgent
from Agents.MedicAgent import MedicAgent

from Generators.VAE import VAE

from utils.Blackboard import Blackboard
from utils.ScoringEngine import ScoringEngine
from utils.utils import load_property_config, extract_hard_filter_keys, extract_soft_filter_keys, update_vae_backbone

PROPERTY_CONFIG = load_property_config("configs/PropertyConfig.yaml")
PATH_CONFIG = load_property_config("configs/PathConfig.yaml")

NUM_GENERATIONS = 10
STEPS_PER_GENERATION = 100

def main():
    vae = VAE(model_path="models/vae_model.pt") # TODO: Update model path. Take from config.
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

    print(f"Starting Discovery Pipeline: {len(agents)} agents initialized.")

    for gen in range(NUM_GENERATIONS):
        print(f"\n--- STARTING GENERATION {gen} ---")
        for step in range(STEPS_PER_GENERATION):
            for agent in agents:
                agent.run_step()
        
        num_hits = len(blackboard.hall_of_fame)
        tasks_left = sum(len(q) for q in blackboard.task_queues.values())
        print(f"Generation {gen} Stats:")
        print(f" - Successful Candidates (Hall of Fame): {num_hits}")
        print(f" - Unsolved Tasks on Blackboard: {tasks_left}")

        if num_hits >= 50:
            print("The Council is meeting: Fine-tuning VAE Backbone on successes...")
            successful_zs = [item[0] for item in blackboard.hall_of_fame]
            with open(PATH_CONFIG['successful_molecules_path'], 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for item in blackboard.hall_of_fame:
                    writer.writerow([item[1], item[2]])
            
            update_vae_backbone(vae, successful_zs)
            blackboard.hall_of_fame = [] 

    print("\n--- PIPELINE COMPLETE ---")
    print(f"Total Unique Drug Candidates Found: {len(blackboard.hall_of_fame)}")

if __name__ == "__main__":
    main()