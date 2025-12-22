
<p align="center">
  <img src="MultiAgent.png" width="100%" />
</p>

# Multi-Agent Drug Design (MADM)

This is a research-oriented skeleton repository for a multi-agent,
property-aware drug design framework.

Components:

- Multiple molecule generators (Generator A–D) coordinated by a
  cross-communication loss that encourages diverse yet useful samples.
- Property validating agent computing QED, SA, docking score (via a
  learned regressor), and a placeholder for ADMET.
- Pretrained activity classifier that consumes ECFP fingerprints plus
  the four properties to output Active / Inactive.
- RL-style training loop where generators are rewarded for molecules
  predicted to be active and satisfying property constraints.

This repo is a starting template – you should plug in your own data,
generator architectures, and training scripts.
