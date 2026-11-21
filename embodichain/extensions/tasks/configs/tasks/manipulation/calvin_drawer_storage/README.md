# CALVIN drawer storage

A Franka FR3 with its parallel hand opens table D’s drawer, picks the red block
from the tabletop, releases it inside, and closes the drawer. The blue and pink
blocks remain in the scene as distractors. CALVIN provides the original
workcell geometry. This deployment adds an EmbodiChain Task Program and an
independently measured final objective. The red block weighs 0.10 kg in this
Franka/PhysX deployment. The original CALVIN file specifies 1 kg.

```bash
embodichain run-task --gym-config embodichain/extensions/tasks/configs/tasks/manipulation/calvin_drawer_storage/task.franka.yaml --headless --disable-sensor --renderer no-render
```

The segment rules require measured rail opening and block lift. The final rule
checks all eight corners of the red block in the moving tray frame, the closed
rail, open fingers and clearance from the block. Closing an empty drawer fails.
Drawer-frame +Y is vertical, and +Z points out of the cabinet. All positions
are in metres. The workcell applies the original/Eden scale of 0.8 exactly once.

[CALVIN table D](https://github.com/mees/calvin_env/tree/797142c588c21e76717268b7b430958dbd13bf48/data/calvin_table_D)
and the three block geometries are credited to Oier Mees and the CALVIN
project (MIT). The versioned asset includes the source license and per-file
provenance. A contact-ready URDF splits the concave cabinet collision into
convex parts and gives the original drawer's grasp bar a named fixed link.
The original visual meshes, objects and source URDFs are retained.

On the CPU PhysX backend with seed 0, the headless expert completed all four
segments in 2,564 control steps. The red block rose by 0.178 m during pickup.
