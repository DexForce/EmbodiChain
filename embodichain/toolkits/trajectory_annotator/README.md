# 遥操轨迹图像序列标注

在浏览器逐帧查看遥操视频、HDF5 相机图像或内存图像序列，为原子技能标注起止帧。支持时间轴定位、技能列表、区间编辑和删除、撤销重做、JSON 自动保存与继续编辑。标注解码使用 CPU；可选关节拟合复用离线 FK/IK，不启动仿真。

![原子技能区间标注界面](../../../docs/architecture/images/trajectory-annotation.png)

## 启动

在已经安装 EmbodiChain 的环境中，安装本工具的依赖即可：

```bash
conda activate py312
python -m pip install 'av>=15.1,<18' 'fastapi>=0.115,<1' 'uvicorn>=0.30,<1'
embodichain annotate-trajectory --video /data/episode_0001.mp4
```

新安装项目时也可以选择 `pip install 'embodichain[trajectory-annotation]'`。源码安装使用 `pip install -e '.[trajectory-annotation]'`，它会同时解析项目的基础依赖；已有开发环境可以使用上面的单独依赖安装命令。

命令会打印本地网址，并默认打开浏览器。首次打开时完整扫描选定视频流、校验来源并建立准确的帧索引；长视频需要等待。按 Ctrl+C 退出服务。

```bash
embodichain annotate-trajectory \
  --video /data/episode_0001.mp4 \
  --annotations /data/labels/episode_0001.segments.json \
  --video-stream 0 \
  --port 8765 \
  --no-browser
```

- `--annotations`：读写同一标注文件，默认视频同目录的 `<stem>.segments.json`。
- `--video-stream`：容器中的实际流索引，默认第一个视频流。
- `--port`：默认 0，自动选择空闲端口；服务只监听 `127.0.0.1`。
- `--no-browser`：手动打开打印的网址。

也可使用 `python -m embodichain.toolkits.trajectory_annotator --video ...` 启动。

## HDF5 输入

```bash
embodichain annotate-trajectory \
  --hdf5 /home/oem/piper-sim-data/label_down/episode_20261022T073344_323022Z/episode.hdf5 \
  --annotations outputs/trajectory_annotation/episode.cam_high.segments.json

# 指定其他相机或缺少元数据时提供帧率
embodichain annotate-trajectory --hdf5 /data/episode.hdf5 \
  --image-key observations/images/cam_left_wrist --fps 30
```

- `--video` 和 `--hdf5` 二选一。HDF5 以只读方式打开，不转成临时视频。
- `--image-key` 默认自动定位 `observations/images/cam_high`，同时兼容单数 `observation/images/cam_high`；两者均存在时要求显式选择。实际 demo 使用复数路径。
- 支持逐帧 JPEG/PNG 字节数组（含变长 uint8 和定长填充行）及 `N×H×W×3 uint8` 原始图像。只按需读选中相机的单帧，初始化时仅解码首帧确认尺寸；文件摘要按 1 MiB 分块计算。
- 原始数组默认 RGB，可用 `--color-order bgr` 指定 BGR；压缩图像按编码解码。
- `--fps` 优先级最高，否则依次读取图像 dataset 的 `fps`、根属性 `fps`、根 `metadata_json.record_fps`。缺少有效值时提示提供 `--fps`，不会猜测。页面明确显示时间由 FPS 推算；帧号始终来自 dataset 行索引。
- 默认标注路径为 `<stem>.<camera>.segments.json`，可用 `--annotations` 覆盖。

[可运行示例](../../../examples/toolkits/trajectory_annotation/README.md)默认使用上述 demo：

```bash
embodichain run-example toolkits/trajectory_annotation
```

## 原子技能参数辨识与连续回填

同一网页的时间轴下方提供拟合面板，复用 `fit-trajectory` 的原生技能重建与连续拼接核心，支持
**Pickup / Slide / Twist / HandOver**。不需要物体 pose、mesh 或仿真；需要与输入机械臂对应的
URDF、TCP、关节顺序、夹爪命令和关节限制配置。配置与关节格式详见
[拟合示例](../../../examples/toolkits/skill_fitting/README.md)。

![网页中的参数辨识、关节误差与 FK 对比](../../../docs/architecture/images/trajectory-annotation-fitting.png)

```bash
embodichain annotate-trajectory --video /data/episode.mp4 \
  --joints /data/joints.npz --robot-profile /data/robot_profile.json \
  --qpos-semantics measured_state --time-offset 0 \
  --annotations /data/labels/episode.segments.json
```

上述拟合参数均可省略，之后在网页填写或修改本机文件路径。`run-example toolkits/trajectory_annotation`
支持同样的拟合参数与 `--video` / `--hdf5`。HDF5 内的关节也可直接使用同一文件：

```bash
embodichain annotate-trajectory --hdf5 /data/episode.hdf5 \
  --joints /data/episode.hdf5 --qpos-key observations/qpos \
  --timestamps-key observations/timestamps \
  --robot-profile /data/robot_profile.json \
  --qpos-semantics measured_state --time-offset 0
```

1. 在右侧下拉菜单选择原子技能并提交区间。DemoSkill 单独置于菜单顶部的“保留原始轨迹”分组，明确标注“不拟合，保留原始轨迹”；选择后编辑区和拟合面板均显示说明，拟合按钮禁用。其余菜单包含项目内置技能，当前只有 Pickup / Slide / Twist / HandOver 支持参数重建。
2. 填写关节记录、机器人配置、数据语义，以及视频零时刻对应的关节时间（秒）。不会从视频 FPS 推断关节采样周期。
3. 点击“拟合并回填选中区间”；页面先保存标注，再在后台执行拟合。拟合期间可继续逐帧查看视频。
4. 查看参数与校验报告，对比原始、原生技能（按源时钟对齐）、连续回填的关节曲线和误差。
   曲线点击按真实帧时间定位视频，支持 VFR；FK 骨架与 TCP 路径随视频更新，可拖动旋转、滚轮缩放并放大显示几何误差。
5. “下载完整回填 NPZ”导出全部原始/重建/回填/校正候选关节、时间戳与元数据。原始视频、HDF5、关节记录均不被覆盖。

完成拟合后可点击“加载机器人模型叠加”，在内嵌 Viser 中查看完整 URDF 机械臂：
原始机械臂为实体，所选轨迹为青色半透明虚影。拖动视频时间轴或播放视频时，
模型按同一物理时间更新关节位置；可在 Viser 控件中隐藏任一模型，或切换回填与原生技能。
模型旁显示当前图像，配套帧进度控件按绝对帧号同步定位图像、全部机械臂和夹爪。
默认查看本次拟合所用的完整标注区间，也可选择拟合核心区间或整段视频；左右按钮逐帧查看，
进度条支持拖动及方向键、Home / End 定位。图像和模型时间仅在最新请求的帧成功解码后更新，
使用实际视频时间戳，支持变帧率输入。
未通过校验时默认显示“校正候选（未通过）”，也可切换到保留原值的连续回填。
“独立窗口”可单独打开同一个模型场景；两处共享当前视频游标。

模型预览需要机器人配置所引用的 URDF 和全部可视 mesh 文件，mesh 路径按 URDF 所在目录解析。
缺少资产时会明确提示，仍可展开“简化 FK 骨架与 TCP 轨迹”查看数值结果。
同一 URDF 的多个机械臂按配置中的基座变换对齐，关节按名称匹配，mimic 由模型处理。
Viser 仅在首次加载模型时启动一个本地服务；替换拟合结果或退出标注服务时一并关闭。
模型更新由一个后台线程持有，视频游标只保留最新待处理时间，不积压播放请求，不启动仿真。

“数据字段、阶段提示与拟合容差”可设置 `qpos` / `timestamps` 字段、显式 `joint_dt`、PT 环境索引、
完整阶段分区 JSON，以及 `SkillFitCfg` 覆盖项。也可用 `--phase-hints` / `--fit-config` 从 JSON 文件预填。
“拟合误差阈值”可选宽松、标准和自定义 JSON。新项目未指定误差配置时默认宽松：
关节 RMSE 为 `0.5`（原始关节单位），TCP RMSE 和阶段位置误差为 `0.1 m`，
姿态 RMSE 和阶段姿态误差为 `1 rad`。标准档使用 `SkillFitCfg` 默认误差阈值。
切换档位会保留过渡时长、候选预算等其他 JSON 配置；修改后重新拟合生效，结果保留实际阈值。
误差档位与关节限位、速度和加速度校验独立，宽松档不会修改机器人配置中的运动限制。
首尾各保留三个原始关节采样，保持边界位置和有限差分导数。默认目标过渡时长为 `0.5` 秒，
使用实际时间戳计算五次过渡，首尾长度独立受技能阶段锚点和选区空间限制，且至少请求 `12` 个采样。
需要更缓的衔接时可填写 `{"blend_duration_s":0.8}`；结果区显示本次过渡配置。
该处理保留原始时钟和区间外采样，仍需通过关节速度、加速度与接触校验。
“原生技能（未拼接）”保留原生解作诊断，其首尾可能跳变；观察衔接效果应选择连续回填或校正候选。
没有可辨识的夹爪开闭或完整阶段提示时会报告失败；RGB 不用于自动判断接触。

标注范围允许包含技能前后的过渡动作。在编辑区间中勾选“单独设置拟合核心区间”，填写标注范围内部的核心起止帧；
标注边界保持不变，网页和 `fit-trajectory --annotations ... --segment-id ...` 均仅重建核心区间，
保留前后过渡采样。没有设置核心区间时使用整个标注范围；核心区间仍须包含完整技能阶段。
不规则示教可标记为 **DemoSkill**（`execute_demo_skill`）并保留原始轨迹；此标记本身不创建可执行 DemoSkill 资产，
也不执行参数重建。旧文件中的自由文本标签仍可读取、编辑，不会自动改名。

每次拟合以输入的原始关节记录为基准，仅替换**选中的一个核心区间**，保留原始时间戳与区间外采样，
多次拟合不会自动累积修改。候选未通过独立校验时仍可查看和下载，`patched_qpos` 保持原始值，
`proposed_qpos` 保留被拒绝的校正候选。标注修订发生变化后旧结果会标记过期并阻止下载，需重新拟合。
输出语义为设计参考，报告中的物理成功与碰撞状态仍为未知/未验证。

后台最多运行一个拟合任务，不排队；帧解码仍由原先的单一线程持有，只读取选区端点的时间，
不重新建立视频索引。服务最多保留一个结果，退出时清理临时 NPZ。网页最多预览约 1000 个关节采样、
128 个关节和 600 个 FK 采样，预览 JSON 上限 16 MiB；下载保留全部采样。
原生规划的时间预算为协作式检查，不能中断单次原生求解；关闭服务会等待正在执行的拟合结束。

## 图像帧输入接口

核心只依赖 [`FrameSource`](frames.py) 协议，不依赖视频或 HDF5。已有 RGB 图像可直接传入：

```python
from embodichain.toolkits.trajectory_annotator.app import launch_annotation
from embodichain.toolkits.trajectory_annotator.frames import ArrayFrameSource

# images: N×H×W×3 uint8 RGB 数组，或同尺寸图像列表
launch_annotation(
    lambda: ArrayFrameSource(images, fps=30, name="episode"),
    "outputs/episode.segments.json",
)
```

调用方保留数组所有权，在应用关闭前保持内容和顺序不变。适配器逐张校验并计算摘要，不额外复制整个序列。
大规模数据可自定义 `FrameSource`，实现 `prepare`、`get_frame`、`check_unchanged` 和 `close`，返回 `FrameSourceInfo` 与单张 `ImageFrame`。
图像统一为 `H×W×3 uint8 RGB`，帧号从 0 开始；借用图像缓冲区至少有效到下一次读帧。
工厂创建、准备、读帧和关闭均在一个工作线程执行；保存线程只调用不访问解码/HDF5 句柄的 `check_unchanged`。
应用负责预览缩放、JPEG 编码、有界缓存和标注保存。程序化服务入口为 `create_annotation_app(factory, annotations)`，启动入口为 `launch_annotation`。
文件适配器分别位于 [`video.py`](video.py) 与 [`hdf5.py`](hdf5.py)，公共接口参考[API 说明](../../../docs/source/api_reference/trajectory_annotator.rst)。

## 标注

1. 用时间轴、帧号输入框或播放按钮定位原子技能起点，左右键逐帧精调。
2. 按 **I** 设置起点，从下拉菜单选择原子技能，例如 Pickup、Twist 或 DemoSkill。选择后按 Enter 返回视频区。
3. 定位原子技能最后一帧，按 **O** 设置终点。
4. 检查原子技能、起止帧和帧数，按 **Enter** 提交。已完成区间显示在时间轴和右侧列表中。
5. 选中列表中的标注，按 **E** 修改、Delete 删除，或点击起止帧跳转。

帧号从 **0** 开始，起止帧**均包含**。`[120, 185]` 包含 66 帧，对应 Python 切片 `frames[120:186]`。允许空隙和单帧原子技能，同一轨道的已完成区间不能重叠。

桌面布局左侧为视频和与视频等宽的原子技能时间轴，右侧依次为“已标注原子技能区间”和“新建/编辑原子技能区间”。标题栏常驻提示 ←/→ 逐帧、空格播放/暂停、I/O 标注起止帧。

整条时间轴均可按住拖动，包括已标注色块内部；游标按鼠标位置定位，不吸附到区间边界。点击色块可同时选中该技能，列表中的起止帧按钮仍用于精确跳转。
每帧占一个等宽单元，闭区间覆盖从起点单元左沿到终点单元右沿；游标和刻度位于对应帧单元中心。色块不会因标签长度、内边距或最小宽度被撑宽，短标签可悬停查看。
长序列在缩略比例下一个屏幕像素可能对应多帧，精确定位可放大时间轴、使用 ←/→ 或输入帧号。

每次只有一个草稿。编辑已有标注时，原版本保留到提交；Esc 取消。删除和撤销只修改标注。存在草稿时先提交或取消，再切换编辑、删除或撤销。

| 快捷键 | 操作 |
| --- | --- |
| Space | 播放 / 暂停，末帧重新播放时从头开始 |
| ← / → | 前 / 后一帧；Shift 跨 10 帧 |
| Home / End | 首帧 / 末帧 |
| G | 输入帧号跳转 |
| I / O | 将当前显示帧设为起点 / 终点 |
| L | 选择原子技能 |
| Enter / Esc | 提交 / 取消草稿 |
| E | 编辑选中的标注 |
| Delete / Backspace | 删除选中的标注 |
| Ctrl/Cmd + Z | 撤销；加 Shift 重做，最多保留 100 次 |
| Ctrl/Cmd + S | 保存已完成标注 |
| ? | 快捷键帮助 |

输入文本时保留正常的文字编辑快捷键。播放中设置端点会暂停在当前已经显示的帧；手动寻帧期间需等画面显示后再标记。页面失去焦点后自动暂停。

## 保存与输出

提交后的修改在停止编辑 1 秒后自动保存。也可点击“保存”或按 Ctrl/Cmd+S。新项目在第一次保存前不创建文件；未提交草稿不写入 JSON。状态栏明确显示未保存、保存中、保存成功或失败。

重新用同一命令打开数据会加载原标注。JSON 使用 `schema_version: 2`（保存核心区间时使用 v3），包含 0 基/闭区间声明、与容器无关的 `source` 信息（类型、身份、帧数、尺寸、帧率、时间来源和适配器详情）、修订号，以及 `segments` 数组。界面与内部模型使用原子技能区间命名；保留现有 JSON/API 的 `segments` 字段和文件名后缀，以继续读取已有标注。数组元素示例：

```json
{
  "id": "seg_2e841dc2-0e4c-4b37-8d97-1d064da7e8a1",
  "label": "pick_up",
  "start_frame": 120,
  "end_frame": 185,
  "note": "可选备注"
}
```

写入前校验，临时文件 fsync 后原子替换；失败时保留旧文件与页面修改。多标签页的旧修订和磁盘外部修改会被拒绝覆盖，可先“导出 JSON”保留当前已完成标注，再重新打开。导出副本不代表本地保存成功。

可选 `core_interval: {"start_frame": 120, "end_frame": 185}` 声明标注内部的拟合核心范围，同样使用绝对视频帧号和闭区间。其前后采样仍属于原标注，回填时保留原值。文件包含该字段时保存为 v3，旧版本读取器会显式拒绝该格式。

文件路径相对于标注 JSON 保存。文件适配器用 SHA-256 绑定源内容，并将相机路径、流索引、帧率或颜色解释等选择纳入身份检查；移动文件后可显式传入新路径，相同内容仍可恢复。内存图像按像素、顺序和 FPS 计算身份。不同内容或选择不能套用已有标注。兼容旧版 `schema_version: 1` 的 `source_video` 文档，读取时只在内存中转换，下次成功保存才写入 v2（含核心区间时为 v3）。

本工具生成区间标注文件。将视频帧映射到机器人控制 step、裁剪视频或切分关节轨迹需要后续消费程序显式处理；现有 recorder 的半开区间不能直接复用本工具的闭区间端点。

## 视频与资源约定

以 MP4/H.264 为主要验收输入，其他容器/编码取决于 PyAV/FFmpeg。支持恒定与可变帧率，以及非零起始 PTS。首版要求帧 PTS 存在且严格递增、视频分辨率固定；原始无时间戳 H.264 流等输入会报告原因。播放不处理音频。

帧号按完整解码的显示顺序定义，包含 B 帧和解码器尾帧，不能由 `time × FPS` 推算。随机跳转按关键帧定位并校验 PTS，必要时从头解码。预览最长边默认 1280 像素；JPEG 缓存默认上限 64 MiB，紧凑帧索引另设 64 MiB 预算，不预加载完整 RGB 视频。性能不足时播放降低实际速度，不跳过索引帧或积压请求。

## 开发验证

```bash
python -m pytest tests/toolkits/trajectory_annotator/test_annotation.py tests/toolkits/trajectory_annotator/test_frames.py -q

# 可选：实际浏览器键盘、删除恢复和异步帧/保存检查。
python -m pip install playwright
python -m playwright install chromium firefox
python -m pytest tests/toolkits/trajectory_annotator/test_browser.py -q
```

浏览器检查也可使用系统 Chrome，或通过 `ANNOTATION_CHROMIUM_EXECUTABLE` 指定可执行文件。缺少可选浏览器时明确跳过。

性能测量入口支持 `--video` 或 `--hdf5`，例如：`python -m scripts.benchmark.toolkits.trajectory_annotation --video /data/episode.mp4`。报告初始化时间、随机寻帧、连续取帧及缓存延迟、RSS 和索引/缓存占用；具体数值取决于分辨率、编码、关键帧间隔和硬件。

实际 HDF5 demo（3644 帧、640×480、30 FPS）在 AMD Ryzen 9 9950X、py312、JPEG 质量 85 上，含文件摘要的初始化为 0.215 秒；50 次随机预览中位数 0.993 毫秒、P95 1.101 毫秒。测量不含 HTTP 和浏览器渲染，不据此保证固定播放帧率。
