# LIBERO stove and moka pot

This EmbodiChain FR3 workcell adapts the
[LIBERO-10 kitchen scene 3 objective](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl).
A hinged knob, the moka pot, and the distractor pan use native collision
geometry. The stove uses an immutable [Hugging Face asset archive](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/753fa474805f6b6d0a7aad196f8f400ea5f32a1a/scene_assets/libero_stove_moka_v1/LiberoStoveMokaV1.zip)
with the primitive source and its provenance. The robot starts at a
calibrated pregrasp pose in the adapted workcell.

The Task Program turns the knob with Twist, then uses Pick and Place to
transfer the pot. Place releases above the burner. The task waits for the
pot to settle and checks the knob's measured angle, the pot's position within
the burner region, and its upright orientation.
