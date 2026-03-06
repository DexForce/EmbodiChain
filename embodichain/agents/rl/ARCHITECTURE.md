# RL训练框架架构

## 总体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                           Trainer                               │
│                    (训练总协调者)                                │
│                                                                 │
│  ┌─────────────────┐        ┌──────────────────┐              │
│  │  初始化阶段      │        │   训练循环        │              │
│  │                 │        │                  │              │
│  │  1. 创建Policy   │───────▶│  while epoch:   │              │
│  │  2. 创建Algo     │        │    ├─ 收集数据   │              │
│  │  3. 创建Collector│        │    ├─ 更新策略   │              │
│  │  4. 创建Env      │        │    └─ 评估性能   │              │
│  └─────────────────┘        └──────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Collector│   │Algorithm │   │  Policy  │
    └──────────┘   └──────────┘   └──────────┘
```

## 核心组件

### 1. Trainer（训练器）
**职责**：总协调者，串联所有组件
```
训练循环：
  for epoch in range(n_epochs):
      ├─ rollout = collector.collect(n_steps)     # 收集数据
      ├─ metrics = algorithm.update(rollout)      # 更新策略
      └─ eval_reward = evaluate(policy)           # 评估性能
```

### 2. Collector（数据收集器）
**职责**：与环境交互，收集经验数据

```
┌─────────────────────────────────────────────┐
│           Collector 类型                     │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────┐  ┌─────────────────┐ │
│  │ SyncCollector    │  │ AsyncCollector  │ │
│  │ (同步收集)        │  │ (异步收集)       │ │
│  │                  │  │                 │ │
│  │ 用于标准RL算法    │  │ 用于VLA模型     │ │
│  │ - PPO           │  │ - 后台持续收集   │ │
│  │ - SAC           │  │ - 独立线程       │ │
│  └──────────────────┘  └─────────────────┘ │
│                                             │
└─────────────────────────────────────────────┘

工作流程：
  obs = env.reset()
  for step in range(n_steps):
      ├─ policy.forward(obs, deterministic=False)  # 采样动作
      ├─ next_obs, reward, done = env.step(action)
      └─ 存储到 TensorDict: (obs, action, reward, done, value)
  return rollout_tensordict  # [T, N] 格式
```

### 3. Algorithm（算法）
**职责**：策略更新逻辑

```
┌─────────────────────────────────────────────┐
│            Algorithm 类型                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │     PPO      │  │     SAC      │  ...   │
│  │              │  │              │        │
│  │ - GAE计算    │  │ - Q学习      │        │
│  │ - Clip损失   │  │ - Soft更新   │        │
│  │ - 价值损失   │  │ - 熵正则化   │        │
│  └──────────────┘  └──────────────┘        │
│                                             │
└─────────────────────────────────────────────┘

工作流程：
  def update(rollout: TensorDict) -> dict:
      ├─ 计算优势函数 (GAE)
      ├─ 多轮优化循环
      │   ├─ policy.evaluate_actions(batch)  # 重新计算log_prob
      │   ├─ 计算loss (clip + value + entropy)
      │   └─ optimizer.step()
      └─ return metrics
```

### 4. Policy（策略）
**职责**：神经网络，输出动作和价值

```
┌─────────────────────────────────────────────┐
│             Policy 类型                      │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │ ActorCritic  │  │  VLAPolicy   │        │
│  │              │  │              │        │
│  │ - MLP网络    │  │ - 视觉语言   │        │
│  │ - 高斯策略   │  │ - 预训练模型 │        │
│  └──────────────┘  └──────────────┘        │
│                                             │
└─────────────────────────────────────────────┘

接口方法：
  1. forward(obs, deterministic=False)
     ├─ 训练时：采样动作 (deterministic=False)
     ├─ 评估时：确定性动作 (deterministic=True)
     └─ 返回：action, log_prob, value

  2. evaluate_actions(obs, action)
     └─ 重新计算给定动作的log_prob和entropy

  3. get_value(obs)
     └─ 仅返回价值估计
```

## 数据流动（TensorDict）

```
Environment ──▶ Collector ──▶ Algorithm ──▶ Policy
    │               │              │            │
    │          TensorDict      TensorDict   Parameters
    │          [T, N]          [batch]      Update
    │               │              │            │
    └───────────────┴──────────────┴────────────┘

TensorDict 结构：
{
  "observation": Tensor or nested TensorDict,
  "action": Tensor[T, N, action_dim],
  "reward": Tensor[T, N, 1],
  "done": Tensor[T, N, 1],
  "value": Tensor[T, N, 1],
  "sample_log_prob": Tensor[T, N, 1],
  "advantage": Tensor[T, N, 1],      # GAE计算后添加
  "return": Tensor[T, N, 1],         # GAE计算后添加
}
```

## 完整训练流程示例

```python
# 1. 初始化组件
trainer = Trainer(
    env=env,
    policy=ActorCritic(...),
    algorithm=PPO(...),
)

# 2. 创建Collector
collector = SyncCollector(
    env=env,
    policy=policy,
    device=device,
)

# 3. 训练循环
for epoch in range(n_epochs):
    
    # 3.1 收集数据
    rollout = collector.collect(
        n_steps=2048,
        reset=True,
    )
    # rollout: TensorDict[T=2048, N=num_envs]
    
    # 3.2 更新策略
    metrics = algorithm.update(rollout)
    # metrics: {"loss": ..., "clip_frac": ..., ...}
    
    # 3.3 评估性能
    eval_reward = trainer.evaluate(
        n_episodes=10,
        deterministic=True,  # 评估时使用确定性动作
    )
    
    # 3.4 日志记录
    print(f"Epoch {epoch}: reward={eval_reward}, loss={metrics['loss']}")
```

## 关键设计原则

### 1. 职责分离
- **Trainer**: 协调者，不涉及具体实现
- **Collector**: 只负责数据收集，不做策略更新
- **Algorithm**: 只负责策略更新，不做数据收集
- **Policy**: 只负责网络前向，不涉及训练逻辑

### 2. 统一接口
- 所有组件使用 **TensorDict** 进行数据传递
- Policy暴露统一接口：`forward()`, `evaluate_actions()`, `get_value()`
- 易于切换不同实现（ActorCritic ↔ VLAPolicy）

### 3. 灵活扩展
- 添加新算法：继承 `BaseAlgorithm`，实现 `update()`
- 添加新策略：继承 `Policy`，实现三个抽象方法
- 添加新收集器：继承 `BaseCollector`，实现 `collect()`

### 4. 确定性评估
```python
# 训练时（随机采样，探索）
policy.forward(obs, deterministic=False)  # 使用 dist.sample()

# 评估时（确定性，稳定）
policy.forward(obs, deterministic=True)   # 使用 dist.mean
```
