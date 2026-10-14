# Franka DFSPH 演示资料

轨迹、参考快照、录像、预览图和验证报告已迁移到
[Hugging Face task assets](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/main/task_assets)，不再随代码仓库提交。

```bash
embodichain data download --name PourCupDemoData
```

```python
from pathlib import Path
from embodichain.data import get_data_path

trajectory = Path(get_data_path("PourCupDemoData/trajectory.pt"))
print(trajectory.parent)  # 同目录包含录像、参考快照及原始验证报告
```

运行时网格单独使用 `PourCupAssets`，任务无需下载演示录像。
资产包提供 MD5 校验，远端 `task_assets/manifest.json` 记录每个文件的 SHA-256。
`particles_30.npz` 仅为历史最终状态参考，不用于初始化水体；默认水体仍在内存中采样。

报告保留原录制环境和提交来源；它们不是新版本完整复跑的声明。
运行方式见[任务 README](../../../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/README.md)。
