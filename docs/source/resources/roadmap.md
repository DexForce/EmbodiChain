# Roadmap

Currently, EmbodiChain is under active development. Our roadmap is as follows:

- Simulation:
    - Rendering:
        - Improve ray-tracing backend performance and fix some konwn issues.
        - Add a high performance Hybrid rendering backend for better visual quality and speed trade-off.
        - Support a more efficient real-time denoiser.
        - Add a new rasterization backend for basic rendering tasks.
    - Physics:
        - Improve GPU physics throughput.
        - We are working on research and development of next-generation physics backend for more accurate and faster simulation.
    - Sensors:
        - Add more physical sensors (eg, force sensor) with examples.
    - Motion Generation:
        - Add more advanced motion generation methods with examples.
    - Useful Tools:
        - We are working on USD support for EmbodiChain to enable better asset management and interoperability.
    - Robots Integration:
        - Add support for more robot models (eg: LeRobot, Unitree H1/G1, etc).

- Data Pipeline:
    - We will release a simple Real2Sim pipeline, which enables automatic data generation from real-world data.
    - We will release an agentic skill generation framework for automated expert trajectory generation.
    - Add simple assets and scenes generator and the integration with data pipeline.

- Models & Training Infrastructure:
    - We will release a modular VLA framework for fast prototyping and training of embodied agents.
    - Add a simple online data streaming pipeline for model training.

- Tasks:
    - Add more benchmark tasks for EmbodiChain.
    - Add more tasks with reinforcement learning support. 
    - We will release a set of tableware manipulation tasks for demonstration of data generation pipeline.
    