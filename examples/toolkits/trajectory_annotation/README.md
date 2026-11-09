# 遥操轨迹原子技能标注示例

在仓库根目录运行，默认绑定 `127.0.0.1`，自动选择端口并打开浏览器。服务运行期间保留终端，结束时按 Ctrl+C。

```bash
conda activate py312
# 若尚未安装网页端依赖：
python -m pip install -e '.[trajectory-annotation]'

# 使用本机提供的 episode.hdf5，无需另外传入参数
embodichain run-example toolkits/trajectory_annotation
```

默认读取 `/home/oem/piper-sim-data/label_down/episode_20261022T073344_323022Z/episode.hdf5`，
自动定位实际的 `observations/images/cam_high`，共 3644 帧、640×480，从 `metadata_json.record_fps` 读取 30 FPS。
标注保存到 `outputs/trajectory_annotation/episode.cam_high.segments.json`；原始 HDF5 只读。
首次操作前不会创建标注文件，再次运行会恢复已有标注。

通过可选参数指定 HDF5 路径和 JSON 输出文件夹：

```bash
embodichain run-example toolkits/trajectory_annotation \
  --hdf5 /path/to/episode.hdf5 \
  --output-dir /path/to/annotations
```

- `--hdf5`：可选，默认读取上述本机 demo。
- `--output-dir`：可选，默认 `outputs/trajectory_annotation`；首次保存时自动创建目录。
- JSON 文件名自动使用 `<HDF5文件名去后缀>.<相机名>.segments.json`。例如 `episode.hdf5`、`cam_high` 对应 `episode.cam_high.segments.json`。
- `--annotations`：也可直接指定完整 JSON 路径，与显式传入 `--output-dir` 二选一。

可进一步指定相机和帧率：

```bash
embodichain run-example toolkits/trajectory_annotation -- \
  --hdf5 /path/to/episode.hdf5 \
  --image-key observations/images/cam_high --fps 30 \
  --output-dir outputs/trajectory_annotation/my_labels
```

也可以直接使用正式 CLI：

```bash
embodichain annotate-trajectory \
  --hdf5 /home/oem/piper-sim-data/label_down/episode_20261022T073344_323022Z/episode.hdf5 \
  --annotations outputs/trajectory_annotation/episode.cam_high.segments.json
```

操作流程：←/→ 逐帧，Space 播放/暂停，I 标起点，O 标终点，填写原子技能名称后 Enter 提交。
右侧显示全部区间，点击帧号跳转，选中后 Delete 删除，Ctrl+Z 撤销，Ctrl+S 保存。
帧号从 0 开始，起止均包含；例如 `[100, 150]` 包含 51 帧。HDF5 的时间显示按 FPS 推算。

## 输入已有图像数组

```bash
# 独立演示：90 张 RGB 图像，不依赖本机 HDF5 demo
embodichain run-example toolkits/trajectory_annotation/images
```

接入已有代码只需一个帧源工厂：

```python
from embodichain.toolkits.trajectory_annotator.app import launch_annotation
from embodichain.toolkits.trajectory_annotator.frames import ArrayFrameSource

# images: N×H×W×3 uint8 RGB 数组，或同尺寸 RGB 图像列表
launch_annotation(
    lambda: ArrayFrameSource(images, fps=30, name="my_episode"),
    "outputs/my_episode.segments.json",
)
```

调用方须在应用关闭前保持数组不变。大规模或按需读取的数据可实现
[`FrameSource`](../../../embodichain/toolkits/trajectory_annotator/frames.py) 协议，返回单张 `ImageFrame`；
网页、预览缓存和标注存储均不依赖 HDF5 或视频格式。
详见[工具说明](../../../embodichain/toolkits/trajectory_annotator/README.md)。
