# Run-agent rect-light intensity override

## Goal

Prevent overly bright rectangular direct lights in vectorized action-agent
runs by reducing their configured intensity to `10.0` whenever more than one
environment is requested.

## Design

Extend `_modify_gym_config_for_run_agent` in
`embodichain/gen_sim/action_agent_pipeline/cli/run_agent.py`. For configurations
where `num_envs > 1`, iterate over `gym_config["light"]["direct"]` and set
`intensity` to `10.0` for every entry whose `light_type` is `"rect"`.

The mutation runs before `build_env_cfg_from_args` parses the configuration, so
the instantiated environment receives the lowered intensity. Missing or
malformed optional light sections are treated as having no matching lights.

## Scope and validation

Only direct rect lights in multi-environment runs change. Single-environment
runs and all non-rect light types retain their configured intensity. Add
targeted unit tests alongside the existing run-agent configuration-modifier
tests for both cases.
