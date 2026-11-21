---
orphan: true
---

# Neural dynamics

See {doc}`/features/neural_dynamics` to compare NeRD, RWM, RigidFormer and
WorldParticle. The first three have shared supervised-training recipes;
WorldParticle currently provides frozen forward inference only.

- {doc}`/features/neural_dynamics/nerd`: state/effort history, pinned Cartpole/Ant
  resources, prediction, training/export, learning-env policy training and
  optional native solver deployment.
- {doc}`/features/neural_dynamics/rwm`: robot state increments, controlled
  fitting, differentiable prediction and replay.
- {doc}`/features/neural_dynamics/rigidformer`: rigid geometry prediction,
  MOVi resources, engine demo, training and evaluation.

- {doc}`/features/neural_dynamics/worldparticle`: fixed inputs, operator setup,
  standalone prediction, optional native solver deployment and four demos.

The {doc}`/api_reference/model_catalogs` page and linked RWM API describe
existing public interfaces. Research protocols remain with the examples.
