"""High-level placeholder script for RL training of the generators.

This does NOT implement a specific generator architecture; you should
subclass BaseGenerator and plug instances into the controller.
"""
import torch
from madm.generators import MultiAgentGeneratorController
from madm.properties import DockingRegressor, PropertyValidatingAgent
from madm.classifier import ActivityClassifier
from madm.rl.rl_loop import rl_training_step

def main():
    # TODO: create actual generator instances implementing BaseGenerator
    generators = []  # [GenA(), GenB(), ...]
    if not generators:
        print("No generators defined - please implement your generator class.")
        return

    controller = MultiAgentGeneratorController(generators)

    docking_model = DockingRegressor()
    classifier = ActivityClassifier()

    prop_agent = PropertyValidatingAgent(docking_model=docking_model)

    params = []
    for g in generators:
        params += list(g.parameters())
    params += list(docking_model.parameters())
    params += list(classifier.parameters())

    optimizer = torch.optim.Adam(params, lr=1e-4)

    for step in range(1, 1001):
        loss = rl_training_step(
            controller,
            property_agent=prop_agent,
            classifier=classifier,
            optimizer=optimizer,
            batch_size=32,
            cross_comm_weight=0.1,
        )
        if step % 10 == 0:
            print(f"Step {step}: RL loss={loss:.4f}")

if __name__ == "__main__":
    main()
