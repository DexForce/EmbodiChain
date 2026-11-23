# PolaRiS Pan Clean

A DROID Franka Panda with a Robotiq 2F-85 operates in the original pan clean capture by Arhan Jain and the PolaRiS team. The instruction is “Use the yellow sponge to scrub the blue handle frying pan”. [Source capture](https://huggingface.co/datasets/owhan/PolaRiS-Hub/tree/8c7e4103e266ef83d8b1ad2e9a63116edd5f155b/pan_clean)

The scene contains `bell_pepper`, `blue_mug`, `coke`, `frying_pan`, `ketchup`, `mustard`, `sponge`, `sushi`. Reach the yellow sponge, lift it more than 3 cm off the table and release it over the frying pan. This measures placement; the rubric has no scrubbing-motion criterion.

The [shared environment guide](../README.md) covers the physical scoring thresholds, action and camera contracts, evaluation entry point, and differences from the published setup.
