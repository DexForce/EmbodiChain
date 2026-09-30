# GenSim Task Program Integration

This package owns GenSim declarations and provider assembly, not another
Task Program executor. The shared compiler, semantic runtime, execution
sessions, Gym bridge, and demo executor remain authoritative.

The execution/compiler implementation follows the `2620929c` baseline. The
limited public additions are optional `SlideGoal.joint_target` and strict decoding
of the existing empty `MoveEndEffectorOptions`. E3, E5 and a
single-object E2 have passed fresh physical regressions against it. The full
can task and final multi-seed qualification remain separate gates. See
`BASELINE_LIMITS.md` for the independent recovery gap and the registered-call
phase-protection gap. The latter blocks complete qualification even when a
normal physical rollout succeeds; terminal acceptance is not in-flight safety.

## Local Boundary

- `assembly.py` composes the existing simulation integration with task-owned
  named stability presets. Its factory only selects provider instances and
  returns the exact public `TaskProgramEnvironmentAdapter`.
- `grasp_filter.py` wraps the existing parallel-jaw sampling service. Immutable
  geometry rules filter unrestricted single-arm proposals by object-local
  region and release clearance. Stock PickUp still chooses symmetric variants,
  solves candidate IK, builds trajectories, and publishes measured effects.
- `stability.py` implements `SegmentPostPolicyPort`. It observes object poses
  and object-to-endpoint drift while yielding ordinary target-qpos holds.
  Only Gym consumes those commands and advances simulation.
- `align_held.py` binds a declared object axis using the latest scene
  observation. It emits a normal `MoveHeldObject` goal with a minimal rotation;
  it neither solves IK nor overwrites the runtime's measured grasp state.
  The explicit `current_object_pose` selector changes orientation without
  moving the object to a nominal scene position. E1 upright placement reuses
  E2's constrained acquisition and staging recipe. E4 upright transfer aligns
  on the source arm before HandOver and corrects on the receiver afterward.
- `stack_place.py` registers a separate stack-placement call and option preset.
  It reuses the original Place goal/effect binding but does not inherit the low
  table-placement TCP cap, which could otherwise eliminate the release retreat.
- `release_clearance.py` binds generated staging heights and a fingerprinted
  withdrawal distance to the existing `MoveEndEffector` skill. The free hand
  first clears vertically, then moves toward its own arm root before Park.
  It refuses a live held attachment and never solves IK or applies commands.
- `motion.py` supplies a planning policy for multi-waypoint EEF approaches.
  Cartesian samples are solved by the original motion generator; command
  timing, execution, cancellation and recovery remain owned by the core.
- `articulation_binding.py` inspects and fingerprints a single-prismatic USD and
  generates explicit Slide/withdraw/Park calls. The E6 Park call is task-owned
  so cleanup preserves the operated arm posture without changing shared Park
  semantics. `articulation_slide.py` binds these to existing public skills and
  measures joint retention via post-policies.
  It does not add a private executor or claim contact-qualified motion.
  Initial table penetration over 2 mm or a buried handle fails before publication
  and at runtime binding. Native joint limits must match the scaled declaration;
  stale public limits are refreshed during assembly using the already-active
  native values, before the first scene observation.
  Scene geometry is measured in the base-link frame that runtime reset preserves.
  Proxy-envelope fitting is an explicit scene-revision operation, never a hidden
  loading fallback or a change to the source USD.
- `constraints.json` is a closed, versioned, inert bundle artifact. The
  integration fingerprint includes its complete contents and the program.
  Unknown fields and old bundles without this artifact fail before execution.
- A policy records measurements and time-window results. It never modifies
  `TaskState`, row eligibility, the program counter, recovery queues, or final
  task success. The core bridge combines its result with the skill result.

`wait_stable` retains its declared meaning: an entity must satisfy a named
stability preset. Upright and stacking presets require a contiguous stable
window. Held-object presets additionally fail on observed grasp slip or target
loss instead of restarting the observation window after a drop.

Placement acceptance is also repeated at the final cleanup segment. A can
passing an early stable check must remain upright after hand clearance and
Park; a stack must remain supported for another complete stable window after
withdrawal. These are ordinary shared segment post-policies and validators,
not a separate task-result update path.
Stack stability measures translation and rotation of both objects against one
shared window. Movement of either object restarts the complete observation
window; a stationary upper object cannot conceal motion of its support.

Coordinated placement additionally checks the scene-generated destination;
being stationary at a wrong location is not sufficient. Coordinated hold
checks that destination and both attachments. Upright HandOver hold checks
the requested axis and the receiving attachment without inventing a position
goal. Ordinary position validators, including Pour return targets, are also
rechecked after cleanup. Upright acquisition and staging use the canonical
`PreparedScene.table_top_z`, not a second required copy in AABB metadata.

The planner's predicted holder map only selects declarations (for example,
omitting a duplicate Pick before a continuation HandOver). It is not runtime
ownership: the shared compiler and runtime still require measured attachment
evidence and block calls after failure.

Baseline registered lowerers may not return action options. GenSim therefore
binds each constrained Pick's declared approach direction into a task-scoped
`gen_sim.pick.<task-group>` policy alias before compilation. Directions come
from scene declarations and preceding planned alignment, not runtime option
mutation. The factory and lowerer identity classes specialize only this inert
call ID because the baseline requires class-level identities; they introduce
no new execution behavior or Atomic Skill implementation.

Each invocation uses the baseline's single shared world direction. Geometry
filtering and stock IK remain row-local. A rejected or empty proposal row uses
the sampler's existing infinite-cost failure convention, never successful
padding or a private eligibility update. End-specific, best-grasp and paired
grasp protocols retain their baseline behavior; constrained Pick requires
unrestricted sampling without a fixed grasp or post-sampling rotation.

The coordinated Robotiq recipe admits 5 mm contact pairs and uses the measured
65 mm pad extent around the configured TCP. The generic 130 mm conservative
envelope is retained for inclined single-arm grasps because the current
three-box collision approximation couples palm position to finger length.
Both service declarations are fingerprinted. Physical joint limits, hand
commands, and contact/holding acceptance thresholds are unchanged.

Stacking from a low handover grasp includes explicit staging, free-arm park,
upper regrasp and final placement calls. The shared Task Program observes every
transition. This is a compiled recipe, not a private retry/state machine.

## Current Migration Status

`configured.py` owns task service decoding and delegates public option decoding
directly to the baseline. It does not inject extra fields into public Options.
`services.py` owns immutable Pick, MoveHeldObject, relative Place and coordinated
transport routes and lowerers. Baseline component composition, common scalar
decoders, Park and Pour are reused without adding fields to the shared YAML
schema. The shared configured service modules and official pour-water format
have been restored to the baseline.

The public language/compiler/validator stack has also been restored: GenSim
upright and stability requirements now use the existing named post-policy port,
not new public validator types. Earlier broad shared Atomic Skill and compiler
changes were withdrawn; the optional Slide joint target is a separate extension.
MoveHeldObject now binds one exact target; alternative
declarations are rejected instead of being silently discarded. Public
HandOver owns its original release sequence without an extra settle phase.
Simulation-factory internals used during assembly are isolated compatibility
dependencies; no Session or Bridge method is replaced.

The fingerprint manifest uses `semantic_integration_fingerprint/v2` with the
explicit `gen_sim.task_program/2620929c/v3` adapter contract. Old bundles are
rejected before component loading and must be regenerated.
Version 3 adds the support object's stable window and a final failed-attempt
camera fetch before reset. The v2 physical runs are retained as historical
evidence, not qualification of this revised acceptance behavior.

E1-E6 and standalone E9 are the Task Engine execution scope. E7/E8 routing is rejected before
semantic graph generation and bundle asset writing. Initial E6 support requires
one environment and a fixed-base, single-prismatic, self-contained metre-authored
USD with uniform scale, a zero closed endpoint, and one identifiable collision
handle. Opening and closing select absolute native joint endpoints, not a fixed
movement independent of current state. A one-second window must stay within
1 mm joint spread and the target tolerance (opening at most 3 cm, closing at most
5 mm, both capped at one quarter of the joint span), after Slide, withdrawal, and
Park. Withdrawal also requires measured hand-open posture. This proves neither
contact-supported sliding nor geometric hand separation or in-flight safety.
Arbitrary scenes and robots are not physically qualified by these checks.

### E9 正式接入

正常入口是 `python -m embodichain.gen_sim.task_engine prepare/run-all/run`。
E9 使用 `gen_sim.press_prepare`、`gen_sim.press`、Park 三个显式段落，沿用
Task Program/Gym 执行，不创建第二执行器。部署必须保留释放初态检查、按压
事件检查及 Park 后的撤离检查；缺少其中任何一个都不能加载。

绑定仅支持 fixed-base、直接连接根刚体的 prismatic button、米制自包含 USD
和均匀缩放；当前只支持一个环境、一个独立 E9 任务，不隐式扩展混合配方。
`press_calibration.json` 记录生成部署的缩放限位、运动部件质量/惯量校准及释放偏置。
这是显式实验物理校准，不冒充源资产原有的弹簧或自锁机构。源 USD 和共享物理默认值不变。
不根据 STOP/emergency 名称生成自锁驱动；接触传感器仅观察，不驱动按钮。
正常重力、速度检查和准确的手指/按钮接触身份均保留。手爪闭合后的几何前缘
用于补偿 grasp TCP，避免把抓取 TCP 当作按压指尖。
按钮、外壳和指垫保留声明或原生接触参数，不覆盖 contact/rest offset。
正常 prepare/run-all 会将绑定的 E9 按钮总成 body_scale 设为 [1, 1, 1]，
保持 XY、朝向和原支撑底面；其他物体不变，源 USD 不覆盖。
press_adaptation.json 记录原始/部署尺度与位置；越出桌面或与其他物体包围盒重叠时拒绝生成。
质量和释放驱动沿用单位化之前的场景校准，避免单纯改变尺寸时同时改变按压阻力。
这是显式生成部署适配，不代表源资产原样通过，也不是按夹爪自动估算尺寸。
E9 物理配置指纹使用资源内容而非绝对存放路径，事务发布及 final/bundle 复制不会改变身份；
修改物理参数或资源内容仍会触发指纹拒绝。旧 E9 bundle 需要重新生成。
显式 route 可声明 `extra_press_distance`（0--0.010 m，默认额外 10 mm）。
它仅增加公共 Press 的命令下压量，不改变按钮目标、限位、物理参数或验收阈值，
并纳入部署指纹。50/100 mm 越程实验不再作为可用部署配置。
这个参数上限不代表目标具有相应碰撞余量，也不构成硬件安全认证。
成功必须包含释放初态、press 阶段连续真实接触支持的至少 0.25 mm 位移，
不能由 Runner 完成、被动下沉、已有按下状态或空接触替代。
不要求按到完整机械端点；按压事件在回弹和撤离后保留，不要求最终持续按下。
普通 Task Program 报告与 `press_evidence.json` 分别记录调用结果及物理子步证据；
记录按钮特写与全景视频。成功含义仅为 `contact_press_0.25mm`，不宣称功能激活或自锁。
旧独立 probe 入口已移除，所有运行验收使用正常 Task Engine。

Physical acceptance requires the user's complete can-stacking and original
tray-holding CLI runs, with measured stable end states and recorded failures.
Provider-level CLI probes are necessary but do not substitute for these runs.
