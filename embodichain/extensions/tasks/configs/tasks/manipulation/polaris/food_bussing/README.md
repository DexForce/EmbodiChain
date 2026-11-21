# PolaRiS Food Bussing

A DROID Franka Panda with a Robotiq 2F-85 operates in the original food bussing capture by Arhan Jain and the PolaRiS team. The instruction is “Put all the foods in the bowl”. [Source capture](https://huggingface.co/datasets/owhan/PolaRiS-Hub/tree/8c7e4103e266ef83d8b1ad2e9a63116edd5f155b/food_bussing)

The scene contains `battery`, `blue_cup`, `grapes`, `ice_cream`, `yellow_bowl`. Reach, lift the ice cream and the grapes more than 6 cm off the table and release each over the yellow bowl. Each food has an independent chain.

The [shared environment guide](../README.md) covers the physical scoring thresholds, action and camera contracts, evaluation entry point, and differences from the published setup.
