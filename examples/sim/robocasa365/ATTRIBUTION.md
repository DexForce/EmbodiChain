# RoboCasa365 replay source credits

- **Kitchen scenes and human demonstrations.** RoboCasa Team, including
  Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri and Yuke Zhu.
  [Project and citation](https://robocasa.ai/#citation) ·
  [Dataset registry at the source revision](https://github.com/robocasa/robocasa/blob/7bfc60c9d26c02506d4265cbe0e86b35e6e25d42/robocasa/utils/dataset_registry.py) ·
  [Release terms for assets and data](https://github.com/robocasa/robocasa/blob/7bfc60c9d26c02506d4265cbe0e86b35e6e25d42/README.md#license) ·
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **LightWheel fixtures, props and their textures.** LightWheel AI
  (model collection), NVIDIA (MJCF release), RoboCasa Team (asset packs).
  [Kitchen MJCF release](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF/tree/420a04af939c34873e6839a586b70844baf28aab) ·
  [RoboCasa asset credits](https://robocasa.ai/docs/build/html/assets/objects.html) ·
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Objaverse models, legacy fixtures and kitchen textures.** The RoboCasa
  Team publishes these packs, including its Midjourney-generated textures,
  under CC BY 4.0. Some individual Objaverse models, the legacy light switch
  and some standard textures still need original-author attribution checks
  before redistribution.
  [Pinned release and license](https://huggingface.co/datasets/robocasa/robocasa-assets/tree/1b92c3d02ca4354984fec961357db0bff7b32166) ·
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **RoboSuite PandaOmron model sources.** Copyright (c) 2022 Stanford Vision
  and Learning Lab and UT Robot Perception and Learning Lab. The original
  Omron CAD provenance still needs verification before redistribution.
  [Pinned model assets](https://github.com/ARISE-Initiative/robosuite/tree/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/models/assets) ·
  [MIT license notice](LICENSES/RoboSuite-LICENSE.txt)
- **Franka Panda geometry.** Franka Emika GmbH and Franka Robotics GmbH.
  [Franka description](https://github.com/frankarobotics/franka_ros/tree/ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98/franka_description) ·
  [Apache 2.0 license](LICENSES/franka_ros-LICENSE.txt) ·
  [Upstream notice](LICENSES/franka_ros-NOTICE.txt)

EmbodiChain selects a fixed episode from each LeRobot archive, exports its
initial state and recorded joint and gripper references, and computes a dynamics
bias from the original model. The MJCF scene is converted to USD with serial
joints for the Omron planar base, material compatibility adjustments and source
inertia and contact properties where supported. Texture images are copied, and
distinct images with a shared basename get distinct filenames. Keep these
credits and notices with any adapted copies.
