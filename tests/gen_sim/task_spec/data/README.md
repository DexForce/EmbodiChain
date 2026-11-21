# Evidence fixtures

`condition_group_cases.json` combines the orientation/count and articulation
fixtures. `e1_count_inside_source_reference.json` covers grouped placement.
Both are projections of the original source captures retained in Git history:
TaskSpec declarations, complete programs, consumed measurements and retained
expected outcomes are unchanged. Unread candidate/role dumps, duplicate geometry
and preparation logs are omitted. Numeric arrays stay readable on compact lines. Identical execution traces are
stored once per declaration and copied into each case by the test loader.

Orientation cases cover aligned, reversed, sideways and missing measurements.
The shared count cases cover failed and missing execution prefixes, so those
cross-products are not repeated for every orientation. Articulation cases keep
endpoint success, a measured negative and a missing joint; its two duplicate
post-execution variants are covered by the generic count-prefix cases. Existing single-binding snapshots protect their published identities. Historical
control snapshots follow their Lab integration tests.

Raw control archives and v1 compatibility snapshots live under
`tests/gym/envs/task_program/evidence/data/` with their protocol tests. TaskSpec
control tests use normalized facts without importing Lab or replaying skills.

The complete-surface cold-import test also decodes group and control bindings
under the same no-numeric/no-Lab import guard, replacing separate subprocesses.
