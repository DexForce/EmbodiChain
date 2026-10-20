# new3 独立 Pick 摸底

固定20个当前场景，每场景事前指定3个不同刚体UID，seed0各一次，共60次。
清单位于入口的`SELECTION`；直立杯壶策略另由`UPRIGHT`明确列举，不按运行结果换物。

```bash
python -B -m scripts.benchmark.task_program.new3_pick prepare
python -B -m scripts.benchmark.task_program.new3_pick run
python -B -m scripts.benchmark.task_program.new3_pick report
```

只调用现有bundle生成器、Default/CPU/B1执行器和串行资源监督。
需要已有归档控制器；不调用生成模型，不建立第二个执行器。
每个对象按自己的用途、手臂和负载完成生产配置，再冻结独立episode。
源场景、位置、尺寸、质量、摩擦和障碍不变；本轮不启用自适应缩放。

验收要求原生双指有效接触、获取门通过、Pick完成、末帧抬升至少8cm，
且末0.20秒保留接触及至少8cm高度。明确直立杯壶还需30度抓型合格。
这里只测既有Pick抬升与末段保持，不包含额外多秒静止持物、后续运输、
放置或倾倒，不认证完整E1/E3，也不认证托盘载物稳定。

源码、输入及视频/回执有哈希。发现源码漂移或资源中断即停止，不自动补考。
验收缓存绑定采集器源码哈希；旧缓存不覆盖，新规则的重评另存`evaluations/`。
`r3`因复用第一个对象的力度配置而排除，8份原生回执保留；`r4`为正式版本。
源文件哈希去重不等于独立几何种类，不能仅凭总成功率声称泛化修好。
