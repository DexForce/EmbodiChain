# MCP URDF assembly showcase

This showcase composes a Universal Robots UR5 arm and a DH PGC 140-50-M
gripper through the project-level MCP server. The composition step uses the
standalone `URDFAssemblyManager`; it does not start a simulator. The optional
verification step loads the generated URDF into a temporary headless
`SimulationManager` world and destroys that world before returning.

## Run it

Install the MCP extra and register the local stdio server:

```bash
python -m pip install -e '.[mcp]'
codex mcp add embodichain -- python -m embodichain mcp
```

Run the complete workflow from the repository root:

```bash
python examples/mcp/urdf_assembly_demo/run_demo.py
```

Use `--no-verify` to exercise only asset resolution, XML assembly, topology
validation, and MCP resources:

```bash
python examples/mcp/urdf_assembly_demo/run_demo.py --no-verify
```

The resolver accepts registered data-asset identifiers instead of arbitrary
filesystem paths. The default catalog includes:

```text
UniversalRobots/UR5/UR5.urdf
DH_PGC_140_50_M/DH_PGC_140_50_M.urdf
```

The first run may resolve these assets through EmbodiChain's normal data cache.
Generated files are written below `outputs/mcp/urdf/` and include copied mesh
assets, the merged URDF, a SHA-256 digest, and an assembly manifest.
The adapter retains a bounded number of assembly directories and removes older
generated handles when the bound is reached.

## Host prompt

An MCP Host can run the same workflow with this request:

```text
Use the EmbodiChain MCP server to compose a modular robot.

1. Call urdf_list_assets and confirm the UR5 arm and DH PGC gripper assets.
2. Call urdf_compose with an `arm` component using
   UniversalRobots/UR5/UR5.urdf and a `hand` component using
   DH_PGC_140_50_M/DH_PGC_140_50_M.urdf.
3. Read the returned manifest and model resources.
4. Call urdf_validate and report links, joints, root link, mesh references,
   and any diagnostics.
5. Call urdf_verify_in_simulation with backend `default` only after pure
   validation succeeds.
6. Report the generated artifact path and the simulation verification result.
```

The MCP boundary therefore exposes a reusable asset workflow while keeping
simulation verification as an explicit provider operation.
