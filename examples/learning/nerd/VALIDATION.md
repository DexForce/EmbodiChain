# NeRD deployment execution record

Checked on 2026-11-15 at implementation commit
`7318c43ad3d9c8e442499af2ac6f1fe0ffb7d407` (#822). Runtime: Linux x86_64,
Python 3.12, Torch 2.11.0+cu128, SDK 0.6.0rc2, Newton 1.7.0, Warp 1.17.0 and
RTX PRO 5000 72GB Blackwell. The process used one CUDA device and two CPU threads.
Published model revision was `78e581526570069410bf269b38c0c52c800b24a7`;
reference assets were `97e202e41fe7fb504656a6b163291305e400780d`.

| Reference system | Native Provider | Environments | Steps | Result |
| --- | --- | ---: | ---: | --- |
| Cartpole | `nerd` | 2 | 60, then one reset replay | Finite state, changed positions, state restore and first-step history replay passed |
| Ant | `nerd` | 2 | 60, then one reset replay | Finite state, changed positions, state restore and first-step history replay passed |

The check used the [native deployment recipe](DEPLOYMENT.md), explicit
`num_substeps=1`, `use_cuda_graph=False` and the export's `warp_reference`
profile. The recorded run applied constant 0.02 effort to both Cartpole
coordinates and Ant's eight actuated coordinates, keeping Ant free-base effort
zero. The quickstart uses cart-only effort for Cartpole; it checks the same
step/reset contract without adding pole effort. Native state was read through
the public articulation batch after every step; no learning-env was created.
The Scene context closed normally. This is execution/ownership evidence, not
balancing, locomotion quality or paper-scale accuracy.

Reproduce from the repository root by preparing the pinned exports and running
both code blocks in [DEPLOYMENT.md](DEPLOYMENT.md). Keep generated resources and
logs outside Git. Validate the current source with these assertions rather
than expecting identical floating point trajectories across runtimes.

The existing focused suite separately covers FK directional finite differences,
repeat backward, old-graph storage safety, native/functional/direct-transition
agreement, reset boundaries and rejection of unsupported dt/base forces. Those
checks are distinct from this published-weight native execution record.

Desktop window presentation and NeRD rendered output were not checked in this
run. Learning-env APG/PPO and analytical policy evaluation have separate
workflows in [README.md](README.md) and the
[policy benchmark](../../../scripts/benchmark/rl/nerd_policy.md); running those
commands does not by itself validate native NeRD deployment.
