# P-01 输入与来源

这些小型手写输入属于实验源材料，后续提交归属为 `scripts/benchmark/tech_reports/assets/physics/P-01/`；现有位置保留以复现旧运行。与原始结果、参考解及录制帧分开管理。

| 文件 | 来源与用途 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| [initial-states.json](initial-states.json) | 本项目生成并冻结的 10 组初态，NumPy PCG64，seed=20261014。 | 559 | `3726a34b0cb28f298d6804b4c3005655e3a26e172597fcc3952df6605aa1d736` |
| [pendulum.urdf](pendulum.urdf) | 本项目按 P-01 质量、惯量、质心和转轴定义编写的无碰撞单摆。 | 848 | `a25b1e1cd43f71afd9904d7ad59fd2d142a60731b5e62ce3bb2d09f1aeff8be8` |
| [pendulum-display.urdf](pendulum-display.urdf) | 由 pendulum.urdf 添加无质量、无碰撞显示杆和颜色；用于同视角录制。 | 1095 | `e6601b1f33c8b1d904d847a9d18800a74f17f789458f8496771f6f65048663eb` |
| [common-view.json](common-view.json) | 本项目编写的共享相机、采样时间、背景刻度和显示几何定义。 | 13037 | `6767e35e4c1024fdf7ae34b869cfcdd8a9e84e96bee6a0e9288f31f21070ea02` |

上述四文件为本项目编写／生成；本地尚未指定它们的公开发布许可。正式发布时在报告 README 为每个文件记录来源与适用许可，不能仅凭 EmbodiChain 的代码许可推定输入许可。此清单不是发布授权或已上传声明。

`physx-source/` 保存分析使用的第三方源码证据，来源见 sources.json（原数据工作区：`assets/P-01/physx-source/sources.json`）。它不属于仿真输入，不进入输入 bundle；保留原许可并在公开打包前核对各文件的再分发要求。

原始运行、有效模型导出和参考数组属于结果；原生 PNG／MP4 属于媒体。它们继续通过合并报告（原数据工作区：`reports/P-01.md`）与素材索引（原数据工作区：`reports/P-01/common-view/index.md`）追溯。
