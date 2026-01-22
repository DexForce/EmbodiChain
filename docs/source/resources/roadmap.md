# Roadmap

Currently, EmbodiChain is under active development. Our roadmap includes the following planned features and enhancements:

- Simulation:
    - Rendering:
        - Improve ray-tracing backend performance and fix some konwn issues.
        - Add a high performance Hybrid rendering backend for better visual quality and speed trade-off.
        - Support a more efficient real-time denoiser.
        - Add a new rasterization backend for basic rendering tasks.
    - Physics:
        - Improve GPU physics throughput.
        - We are working on research and development of next-generation physics backend, supporting high-accuracy simulation, differentiable dynamics, and neural physical models for end-to-end AI integration.
    - Sensors:
        - Add more physical sensors (eg, force sensor) with examples.
    - Motion Generation:
        - Add more advanced motion generation methods with examples.
    - Useful Tools:
        - We are working on USD support for EmbodiChain to enable better asset management and interoperability.
    - Robots Integration:
        - Add support for more robot models (eg: LeRobot, Unitree H1/G1, etc).

- Data Pipeline Coming Soon:
    - We will release a Real2Sim pipeline, which enables automatic data generation and scaling from real-world seeding priors.
    - We will release an agentic skill generation framework for automated expert trajectory generation.
    - Add assets and scenes generator and the integration with data pipeline.

- Models & Training Infrastructure Coming Soon:
    - We will release a modular VLA framework for fast prototyping and training of embodied agents.
    - Add online data streaming pipeline for model training.

- Embodied Tasks Coming Soon:
    - Add more benchmark tasks for EmbodiChain.
    - Add more tasks with reinforcement learning support.
    - Add a set of manipulation tasks for demonstration of data generation pipeline.
    