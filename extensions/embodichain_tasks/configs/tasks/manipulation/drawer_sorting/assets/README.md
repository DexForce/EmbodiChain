# Cabinet asset provenance

`bedside_cabinet.urdf` derives from the project's downloaded
`Drawer/model_split_links_with_inertials.urdf` asset (registered archive MD5
`3981636db1f4188146fce25d54084612`). It retains the complete cabinet and drawer
link topology, material colors, panel collisions, handle, and passive rail.

A uniform 0.8 scale is baked into visual/collision geometry, origin
translations, and prismatic limits/velocity. Link masses are retained and
inertia tensors are multiplied by 0.64. The environment supplies the cabinet's
world pose and supporting plinth. Baking the scale keeps the articulation
geometry adapter's unit-scale mesh/FK contract.
