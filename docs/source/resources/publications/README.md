# Academic Publications

[![DOI](https://img.shields.io/badge/DOI-available-success?style=for-the-badge)](#)
[![Year](https://img.shields.io/badge/year-2025--2026-blue?style=for-the-badge)](#)
---

This page contains bibliographic information for academic papers and competition projects related to EmbodiChain. Entries are ordered by year (newest first).

## Publications

### 2026

#### RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills

**Authors:** RoboSynChallenge Organizing Team

**Venue:** NeurIPS 2026 Competition

**Links:** [Website](https://robosyn-bench.net/#/home) | [Report](https://robosyn-bench.net/material/RoboSynChallenge-report.pdf) | [Code](https://github.com/EDEM-AI/RoboSynChallenge)

```bibtex
@misc{robosynchallenge2026,
  title = {RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills},
  author = {{RoboSynChallenge Organizing Team}},
  howpublished = {NeurIPS 2026 Competition},
  year = {2026},
  url = {https://robosyn-bench.net/}
}
```

---

#### Backpropagating Through Simulation: Analytic Policy Gradients for Sample and Learning Efficient Differentiable Continuous Control

**Authors:** Yueci Deng

**Venue:** arXiv preprint arXiv:2606.21525, 2026

**Links:** [Paper](https://arxiv.org/abs/2606.21525) | [Code](https://github.com/yuecideng/analytic_policy_gradients)

**EmbodiChain integration:** See the [APG algorithm](../../overview/rl/algorithm.md#apg), [differentiable trainer](../../overview/rl/trainer.md#differentiabletrainer), and [training CLI](../../overview/rl/train_script.md) documentation. The Newton-backed [differentiable route](../../overview/sim/sim_manager/physics/newton/runtime_qualification.md#differentiable-route) currently provides a kinematics gradient bridge.

**Examples:**

- **PointMass APG training:** [Training configuration](https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_tasks/configs/tasks/classic_control/point_mass/agents/apg.yaml) and [differentiable environment](https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_tasks/embodichain_tasks/classic_control/point_mass.py). Run the configuration with:

  ```bash
  embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/classic_control/point_mass/agents/apg.yaml
  ```

- **Franka FR3 differentiable kinematics:** [Reference environment](https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_tasks/embodichain_tasks/special/franka_reach_apg.py), demonstrating the Warp-tape / PyTorch-autograd bridge through forward kinematics and reward kernels.

```bibtex
@article{deng2026backpropagating,
  title = {Backpropagating Through Simulation: Analytic Policy Gradients for Sample and Learning Efficient Differentiable Continuous Control},
  author = {Deng, Yueci},
  journal = {arXiv preprint arXiv:2606.21525},
  year = {2026},
  doi = {10.48550/arXiv.2606.21525},
  url = {https://arxiv.org/abs/2606.21525}
}
```

---

#### From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation

**Authors:** Sheng Xu, Ruixing Jin, Huayi Zhou, Bo Yue, Guanren Qiao, Yueci Deng, Yunxin Tai, Kui Jia, Guiliang Liu

**Venue:** Robotics: Science and Systems (RSS), 2026

```bibtex
@inproceedings{xu2026agentchord,
  title = {From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation},
  author = {Xu, Sheng and Jin, Ruixing and Zhou, Huayi and Yue, Bo and Qiao, Guanren and Deng, Yueci and Tai, Yunxin and Jia, Kui and Liu, Guiliang},
  booktitle = {Robotics: Science and Systems (RSS)},
  year = {2026}
}
```

---

#### Sim2Real VLA: Zero-Shot Generalization of Synthesized Skills to Realistic Manipulation

**Authors:** Runyi Zhao, Sheng Xu, Ruixing Jin, Yueci Deng, Yunxin Tai, Kui Jia, Guiliang Liu

**Venue:** The Fourteenth International Conference on Learning Representations (ICLR), 2026

```bibtex
@inproceedings{zhao2026sim2real,
  title={Sim2real vla: Zero-shot generalization of synthesized skills to realistic manipulation},
  author={Zhao, Runyi and Xu, Sheng and Jin, Ruixing and Deng, Yueci and Tai, Yunxin and Jia, Kui and Liu, Guiliang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

---

### 2025

#### DexScale: Automating Data Scaling for Sim2Real Generalizable Robot Control

**Authors:** Guiliang Liu, Yueci Deng, Runyi Zhao, Huayi Zhou, Jian Chen, Jietao Chen, Ruiyan Xu, Yunxin Tai, Kui Jia

**Venue:** Forty-Second International Conference on Machine Learning (ICML), 2025

```bibtex
@inproceedings{liu2025dexscale,
  title={DexScale: automating data scaling for sim2real generalizable robot control},
  author={Liu, Guiliang and Deng, Yueci and Zhao, Runyi and Zhou, Huayi and Chen, Jian and Chen, Jietao and Xu, Ruiyan and Tai, Yunxin and Jia, Kui},
  booktitle={Forty-second international conference on machine learning},
  year={2025}
}
```

---

## Adding a New Paper

To add a new publication:

1. Add a new section under the appropriate year heading
2. Include the paper title, authors, venue, and BibTeX entry
3. Keep entries ordered by year (newest first)

## Core Framework Citations

The following citations are kept in the main [README.md](https://github.com/DexForce/EmbodiChain) as they are considered core framework references:

- **EmbodiChain** - The framework itself
- **GS-World** - The underlying generative simulation paradigm
