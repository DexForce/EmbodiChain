---
orphan: true
---

# Neural dynamics

See {doc}`/resources/models/neural_dynamics/index` to compare NeRD, RWM, RigidFormer and
WorldParticle. The first three have shared supervised-training recipes;
WorldParticle currently provides frozen forward inference only.

- {doc}`/resources/models/neural_dynamics/nerd`: state/effort history, pinned Cartpole/Ant
  resources, prediction, training/export, learning-env policy training and
  optional native solver deployment.
- {doc}`/resources/models/neural_dynamics/rwm`: robot state increments, controlled
  fitting, differentiable prediction and replay.
- {doc}`/resources/models/neural_dynamics/rigidformer`: rigid geometry prediction,
  MOVi resources, engine demo, training and evaluation.

- {doc}`/resources/models/neural_dynamics/worldparticle`: fixed inputs, operator setup,
  standalone prediction, optional native solver deployment and four demos.

The {doc}`/api_reference/model_catalogs` page and linked RWM API describe
existing public interfaces. Research protocols remain with the examples.
