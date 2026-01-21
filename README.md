# EmbodiChain

![teaser](assets/imgs/teaser.jpg)

[![License](https://img.shields.io/github/license/DexForce/EmbodiChain)](LICENSE)
[![Website](https://img.shields.io/badge/website-dexforce.com-green?logo=google-chrome&logoColor=white)](https://dexforce.com/embodichain/index.html#/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-docs-blue?logo=github&logoColor=white)](https://dexforce.github.io/EmbodiChain/introduction.html)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/DexForce/EmbodiChain/releases)
---

EmbodiChain is an end-to-end, GPU-accelerated framework for Embodied AI. It streamlines research and development by unifying high-performance simulation, real-to-sim data pipelines, modular model architectures, and efficient training workflows. This integration enables rapid experimentation, seamless deployment of intelligent agents, and effective Sim2Real transfer for real-world robotic systems.

> [!NOTE]
> EmbodiChain is in Alpha and under active development:
> * More features will be continually added in the coming months. You can find more details in the [roadmap](https://dexforce.github.io/EmbodiChain/resources/roadmap.html).
> * Since this is an early release, we welcome feedback (bug reports, feature requests, etc.) via GitHub Issues.


## Key Features

- 🚀 **High-Fidelity GPU Simulation**: Realistic physics for rigid & deformable objects, advanced ray-traced sensors, all GPU-accelerated for high-throughput batch simulation.
- 🤖 **Unified Robot Learning Environment**: Standardized interfaces for Imitation Learning, Reinforcement Learning, and more.
- 📊 **Scalable Data Pipeline**: Automated data collection, efficient processing, and large-scale generation for model training.
- ⚡ **Efficient Training & Evaluation**: Online data streaming, parallel environment rollouts, and modern training paradigms.
- 🧩 **Modular & Extensible**: Easily integrate new robots, environments, and learning algorithms.

The figure below illustrates the overall architecture of EmbodiChain:

<p align="center">
  <img src="assets/imgs/frameworks.jpg" alt="architecture" width="90%"/>
</p>



## Getting Started

To get started with EmbodiChain, follow these steps:

- [Installation Guide](https://dexforce.github.io/EmbodiChain/quick_start/install.html)
- [Quick Start Tutorial](https://dexforce.github.io/EmbodiChain/tutorial/index.html)
- [API Reference](https://dexforce.github.io/EmbodiChain/api_reference/index.html)


## Citation

If you find EmbodiChain helpful for your research, please consider citing our work:

```bibtex
@misc{EmbodiChain,
  author = {EmbodiChain Developers},
  title = {EmbodiChain: An end-to-end, GPU-accelerated, and modular platform for building generalized Embodied Intelligence},
  month = {November},
  year = {2025},
  url = {https://github.com/DexForce/EmbodiChain}
}
```

```bibtex
@misc{GS-World,
  author = {Liu, G., Deng, Y., Liu, Z., and Jia, K},
  title = {GS-World: An Efficient, Engine-driven Learning Paradigm for Pursuing Embodied Intelligence using World
   Models of Generative Simulation},
  month = {October},
  year = {2025},
  journal = {TechRxiv}
}
```