# 报告 HTML 阅读版

从已有 Markdown、图表、真实仿真原片和数据链接构建静态材料站，不启动仿真。
当前数据上传暂缓，先读 [快照说明](../releases/simulation-report-2026-10-28/README.md)。

```bash
export SIM_REPORT_ROOT=/path/to/report-data-copy
python -m scripts.benchmark.tech_reports.reporting.site.build --output outputs/benchmarks/report-site
python -m scripts.benchmark.tech_reports.reporting.site.serve --directory outputs/benchmarks/report-site --host 127.0.0.1 --port 8766
# 在另一个终端，保持相同的 SIM_REPORT_ROOT：
python -m scripts.benchmark.tech_reports.reporting.site.check --directory outputs/benchmarks/report-site --url http://127.0.0.1:8766
```

依赖见 requirements.txt；KaTeX 0.16.22 及 MIT 许可随源码提供。
修改 catalog.json，勿直接编辑输出 HTML。输出有 13 个实验页；未开展项只保留标题。
publication-manifest.json 记录导出来源、字节数与哈希；check.py 检查链接、状态、哈希、HTTP 和视频 Range 响应。
新建输出目录避免污染已有发布副本。只服务构建目录，不将数据工作区或仓库根目录设为 HTTP 根。
原机器的局域网站和 systemd 配置不由本 PR 安装或替换。
