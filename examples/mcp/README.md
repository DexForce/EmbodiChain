# EmbodiChain MCP discovery

Install the EmbodiChain `mcp` extra and register this stdio command with your
MCP client.

```bash
embodichain mcp --catalog
```

This command lists installed robot presets, packaged task deployments and
physics configuration names before a World exists. The catalog reports
declared deployment metadata. A live Session supplies measured robot state.

```bash
embodichain mcp --catalog --session SESSION_ID --session-directory SESSION_DIR
```

Use `--session-directory` for a custom DexSim discovery directory. The second
command borrows a connection to an existing World. The World owner controls
the simulation lifecycle. Use `embodichain mcp --session SESSION_ID` for just
the original robot-state tool.

| Tools | Result |
| --- | --- |
| `embodichain_server_info`, `embodichain_health` | Package version, transport, selected Session ID and host liveness |
| `embodichain_list_service_capabilities` | Tool names, Resource URIs and templates, and Prompt names registered on this connection |
| `embodichain_list_robots`, `embodichain_get_robot_info` | Registered IDs, variants, declared DOFs and control-part counts |
| `embodichain_list_tasks`, `embodichain_get_task_info` | Qualified task IDs, deployment names, environment IDs and declared capabilities |
| `embodichain_list_physics_backends` | Supported physics configuration names |

New tools advertise structured output schemas with `schema_version: 1`.
Unknown robot and task identifiers return tool errors. The
`embodichain_inspect_setup` Prompt shows how to browse the catalog.

| Resource | Content |
| --- | --- |
| `embodichain://robots/{robot_id}/config` | Declared metadata for the registered default profile. Use `embodichain_get_robot_info` with an explicit variant to inspect another one. |
| `embodichain://tasks/catalog` | Packaged task inventory |
| `embodichain://worlds/<session-id>/manifest` | Bound World identity, DexSim version and configuration, and live Scene lifecycle status |

The manifest URI appears in `resources/list` only for a bound Session. Its
`backend` uses the name in the DexSim host's discovery record. The
`backends` tool lists EmbodiChain configuration names.
