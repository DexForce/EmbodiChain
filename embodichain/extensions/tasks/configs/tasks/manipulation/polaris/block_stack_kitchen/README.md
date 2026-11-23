# PolaRiS Block Stack Kitchen

A DROID Franka Panda with a Robotiq 2F-85 operates in the original block stack kitchen capture by Arhan Jain and the PolaRiS team. The instruction is “Place and stack the blocks on top of the green tray”. [Source capture](https://huggingface.co/datasets/owhan/PolaRiS-Hub/tree/8c7e4103e266ef83d8b1ad2e9a63116edd5f155b/block_stack_kitchen)

The scene contains `green_cube`, `green_tray`, `pink_bowl`, `purple_plate`, `tomato`, `uw_corn`, `wood_cube`. Reach and lift each cube at least 3 cm off the table, release each over the green tray, then release the green cube over the wood cube (50% projected overlap).

The [shared environment guide](../README.md) covers the physical scoring thresholds, action and camera contracts, evaluation entry point, and differences from the published setup.
