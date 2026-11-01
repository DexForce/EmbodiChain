# Legacy Franka Panda selector

`FrankaPandaCfg` is a compatibility selector for the historical FR3 asset family.
`robot_type: panda` continues to choose that model; it now constructs its bare
seven-joint arm ending at `fr3_link8`. New callers use `FrankaCfg` with
`arm_model: fr3` or `arm_model: panda`.

Mount end effectors with ordinary URDF components, control groups, drive
parameters and solver TCP overrides, as for UR robots. The `franka_panda`
embodiment explicitly installs the native hand for existing task deployments.
See [Franka configuration](cfg.md) for arm selection, assembly and migration.

```{eval-rst}
.. py:class:: embodichain_robots.franka.panda.FrankaPandaCfg

   Legacy FR3 arm selector. The inherited ``from_dict()`` and
   ``build_pk_serial_chain()`` use the bare seven-joint arm and flange TCP.
```
