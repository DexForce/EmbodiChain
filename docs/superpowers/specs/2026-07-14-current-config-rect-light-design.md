# Current gym configuration rect light

## Goal

Add one direct rectangular area light to the current action-agent gym
configuration so the tabletop scene has even overhead illumination.

## Design

Replace the empty `light.direct` array in
`gym_project/action_agent_pipeline/configs/current/fast_gym_config.json` with a
single light configuration:

- UID: `main_rect_light`
- Type: `rect`
- Position: `[0.0, 0.0, 2.2]`
- Direction: `[0.0, 0.0, -1.0]`
- Color: white (`[1.0, 1.0, 1.0]`)
- Intensity: `30.0`
- Dimensions: `2.0` wide by `2.0` high

## Scope

EmbodiedEnv already parses direct lights with `LightCfg.from_dict`, and its
simulation backend supports rect lights, including their direction and
dimensions. Therefore this change is configuration-only; no runtime code or
new test coverage is needed. Validation will confirm that the JSON parses and
the configured light instantiates as a rect light.
