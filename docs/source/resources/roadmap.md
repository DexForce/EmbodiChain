# Roadmap

Currently, EmbodiChain is under active development. Our plan for the feature roadmap is as follows:

- Simulation:
    - Rendering:
        - Improve ray-tracing backend performance and fix some konwn issues.
        - Add a high performance Hybrid rendering backend for better visual quality and speed trade-off.
        - Support a more efficient real-time denoiser.
        - Add a new rasterization backend for basic rendering tasks.
        - Support 3DGS rendering mode (If we have enough bandwidth).
    - Physics:
        - Improve soft body simulation stability and add more examples and tasks.
        - We are also exploring how to integrate [newton physics](https://github.com/newton-physics/newton) into EmbodiChain as an alternative physics backend.
    - Sensors:
        - Add contact and force sensors with examples.
    - Kinematics Solvers:
        - Improve the existing IK solver performance and stability (especially SRSSolver and OPWSolver).
    - Motion Generation:
        - Add more advanced motion generation methods and examples.
    - Useful Tools:
        - Add a robot workspace analysis tool for better visualization and sampling of robot accessible workspace.
    - We are working on USD support for EmbodiChain to enable better scene creation and asset management.

- Models and Training Workflows:
