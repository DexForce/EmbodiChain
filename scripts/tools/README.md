# DexTrove 桌面可抓取候选集

[固定目录配置](configs/dextrove/catalog.json)保存了采集时的完整标签字典、所有可见物体的 UUID、名称和原始标签关联。选入抓取候选集的物体还带有唯一的 `category`、包围盒尺寸和筛选依据。分析类别最多 50 个，按物体整体的用途和形态命名，例如水瓶归入 `bottle`，碗归入 `bowl`，勺子归入 `spoon`。每个类别的含义可在配置的 `categories` 字段查询。抓取成败由具体夹具和仿真试验判定。

### 下载固定版本

在可访问 DexTrove API 的 Linux 机器上，从仓库根目录运行。下载依赖 `requests`、`urllib3` 和 OpenUSD 的 `pxr`。普通下载直接读取仓库配置，云端和本地用户均可在没有 Codex CLI 的环境运行。

```bash
conda activate py312
export DEXTROVE_API_ROOT='https://your-dextrove-host/v1'
python scripts/tools/prepare_dextrove_subset.py --output-dir outputs/dextrove_subset
```

用 `--download-limit 1` 可先下载一件；在同一目录去掉数量限制继续运行即可。新下载的源包保存在 `objects/<UUID>/`，会校验服务端给出的大小和 SHA-256。每个候选在 `categories/<category>/<UUID>/<UUID>.usda` 有且仅有一个分类入口。用旧版输出目录运行时会复用 `assets/<原标签>/<UUID>/` 中已有的源包，原先引用这些路径的配置仍可使用。

分析类别分布只需读取 JSON。以下命令可以在离线机器上运行：

```bash
jq -r '.assets[] | select(has("category")) | .category' \
  scripts/tools/configs/dextrove/catalog.json | sort | uniq -c | sort -nr
```

使用七款夹具评测已下载的全部候选物体时，把下载命令的 `--output-dir` 传给 `--dextrove-root`：

```bash
python -m embodichain benchmark grasp-pose-generator run \
  --dextrove-root outputs/dextrove_subset --grid 168 --resume \
  --output outputs/benchmarks/grasp-dextrove-grid168
```

命令逐个检查 `categories/` 中的 UUID，使用 PhysX 导入带碰撞形状的单刚体，再运行抓取。`--grid 168` 表示每批最多放入 168 个夹具与物体组合。候选姿态逐组合生成，每格每轮试抓一个候选；日志里的 `simultaneous grasp trials` 是当轮实际并行数。格数可按机器资源调整，续跑时沿用原格数和结果目录。

结果目录中的 `dextrove_inventory.json` 列出每个物体的类别、是否纳入评测以及排除原因。追加 `--preflight-only` 可以只生成清单并查看预计的组合数。更新数据集后可用新的结果目录保留两次报告。

已有 JSONL 清单可通过 `--shortlist 文件` 重放。每行指定 `asset_id`，工具可从固定配置查出已有 UUID 的类别。自定义行也可以直接给出 `category`。使用另一份已生成的目录配置时，加上 `--catalog 路径`。

### 数据更新时重新分类

维护者使用已登录的 Codex CLI 运行 `--refresh`。默认模型为 `gpt-6.2-astra`。此命令获取最新可见资产、标签和尺寸，从有标签物体中筛选普通桌面上可能被平行夹爪搬取的对象，然后为每个入选 UUID 指定一个分析类别。分类时会读取已有模型的 USD 元数据，将源名称和源类别分别保存为 `usd_name`、`usd_category`。这能帮助识别目录中只有编号的物体。没有本地模型时，分类使用原始名称、标签和尺寸。普通下载用户可直接复用生成的配置。

```bash
python scripts/tools/prepare_dextrove_subset.py --refresh \
  --output-dir outputs/dextrove_refresh

python scripts/tools/prepare_dextrove_subset.py \
  --catalog outputs/dextrove_refresh/catalog.json --output-dir outputs/dextrove_subset
```

刷新默认把新配置写到 `<output-dir>/catalog.json`，并在该目录保存原始采集结果。确认分类后，可将该 JSON 更新到 `scripts/tools/configs/dextrove/catalog.json` 随代码发布。如果已有一轮完整采集，只想重新划分类别，可用下列命令直接复用其标签关联与候选清单：

```bash
python scripts/tools/prepare_dextrove_subset.py --refresh \
  --snapshot-dir outputs/previous_dextrove_snapshot \
  --output-dir outputs/dextrove_reclassified
```
