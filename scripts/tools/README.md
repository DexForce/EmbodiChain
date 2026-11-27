# DexTrove 桌面可交互物体子集

在能访问 DexTrove 服务的 Linux 机器上，从仓库根目录运行。先将示例地址换成你的 DexTrove API 地址，保留末尾的 `/v1`。也可以用 `--api-root` 指定地址。

```bash
export DEXTROVE_API_ROOT='https://your-dextrove-host/v1'
conda activate py312_bk
python scripts/tools/prepare_dextrove_subset.py --output-dir outputs/dextrove_subset
```

工具依赖 `requests`、`urllib3` 和 OpenUSD 的 Python 绑定 `pxr`。从头筛选需要已登录的 Codex CLI，可用 `--model` 指定账号可用的模型，默认使用 `gpt-6.2-astra`。

1. 获取全量可见资产、标签字典与标签关联，按服务端统计校验。根据处理记录的包围盒计算 XYZ 尺寸，单位为厘米；缺少尺寸时保留 `null`。
2. 把有标签物体的名称、标签和尺寸交给模型，初筛可在普通桌面独立放置、可能被最大开口 8～12 cm 的平行夹爪搬取的物体。候选清单标明整体宽度依据或待核实的局部夹持部位。实际抓取结果由后续物理验证给出。
3. 下载通过筛选的源文件，校验字节数与 SHA-256，解包并保留 USD 相对引用。脚本从每个物体的 `tags` 中读取 `domain == "feature.category"` 的英文 `key`，例如 `mug`、`vial`，用作分类目录名。

| 指定目录中的输出 | 内容 |
| --- | --- |
| `tag_dictionary.json` | 标签字典，含英文 `key`、中文显示名与标签域 |
| `asset_tags.jsonl` | 每个物体的完整标签关联，含无标签物体 |
| `objects_tags_size.txt` | 每行一个物体，含名称、标签、`dimensions_cm` 和 `file_size_bytes` |
| `tabletop_grasp_candidates.txt` | 每行一个入选物体，`tags` 保留原始类别标签，`tabletop_category` 记录 Astra 判断的桌面用途 |
| `assets/<英文类别标签>/<asset_id>/<asset_id>.usda` | 可直接打开的 USD 入口，同目录的 `source/` 保存模型及依赖 |

一个物体有多个类别标签时，每个类别目录都能找到它。按英文字典序排第一的类别存放模型，其余类别使用相对符号链接，每件物体只下载一份。有其他标签、但缺少可用英文类别标签的候选归入 `_uncategorized/`。无任何标签的物体在筛选阶段排除。

USD 入口保留源层的 `customLayerData`，供 benchmark 读取 SimReady 声明的 PhysX、Newton 等物理变体。用本工具续跑旧子集时，会从本地 `source/` 更新旧版入口。

如需先试下载一个物体，在命令末尾加 `--download-limit 1`。清单仍会完整生成。去掉数量限制并使用同一输出目录，工具会校验清单、跳过已安装的物体，继续下载其余候选。下载失败时命令返回非零状态。要按 DexTrove 的最新数据重新筛选，可指定新的输出目录。

### 按已有候选清单下载

已有 `tabletop_grasp_candidates.txt` 时，可以在新目录下载清单中的同一批资产 ID。此模式直接从 DexTrove 下载，跳过标签收集和模型调用。清单使用 JSON Lines，每行包含 DexTrove 的 `asset_id` 和 `tags`，可选 `name`。为不同清单使用独立的输出目录可以保留各自的子集。

```bash
python scripts/tools/prepare_dextrove_subset.py \
  --shortlist outputs/dextrove_subset/tabletop_grasp_candidates.txt \
  --output-dir outputs/dextrove_replay --download-limit 1
```
