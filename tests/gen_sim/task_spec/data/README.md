# Frozen evidence provenance

`public_owner_sources.json` records the source-file composition at PR #1091
revision `3c5ec021e9a2ba48c1b135ac5f2abd25b3649ef5`, before protocol ownership
was separated. Its hashes are historical source references, not assertions that
current refactored files remain byte-identical to those paths.

Frozen measurements and expected outcomes retain their original contents.
The HandOver and coordinated raw-protocol oracles now live under
`tests/gym/envs/task_program/evidence/data/`, with their owning integration tests.
Normative TaskSpec tests consume normalized facts and explicit criteria.
