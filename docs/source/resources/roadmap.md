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
        - Improve GPU physics throughput.
        - Improve soft body simulation stability and add more examples and tasks.
        - We are also exploring how to integrate [newton physics](https://github.com/newton-physics/newton) into EmbodiChain as an alternative physics backend.
    - Sensors:
        - Add contact and force sensors with examples.
    - Kinematics Solvers:
        - Improve the existing IK solver performance and stability (especially SRSSolver and OPWSolver).
    - Motion Generation:
        - Add more advanced motion generation methods and examples.
    - Useful Tools:
        - We are working on USD support for EmbodiChain to enable better scene creation and asset management.
        - We will release a simple Real2Sim pipeline, which enables automatic task generation from real-world data.
    - Robots Integration:
        - Add support for more robot models (eg: LeRobot, Unitree H1/G1, etc).

- Agents:
    - Add more Reinforcement Learning examples and environments.
    - We will release a Modular VLA framework for fast prototyping and training of embodied agents.
    - We will release a simple online data streaming pipeline for Imitation Learning.

- Tasks:
    - We will release a set of Real2Sim tasks as examples for EmbodiChain.
    - We will release a set of tableware manipulation tasks for demonstration of data generation pipeline.
    