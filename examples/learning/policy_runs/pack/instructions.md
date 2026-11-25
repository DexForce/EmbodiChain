# Running a registered policy

For tasks covered by the EmbodiChain policy tools, use the operator-created
fresh DexSim World. Query `dexsim_status` for its Session ID. Select the four
versioned robot, policy, scene and task IDs advertised by
`embodichain_prepare_run`. Use the returned plan ID and that same Session ID
with `embodichain_start_run`, then wait with `embodichain_get_run`.

The policy runner owns scene construction and control. Wait with
`wait_seconds=20` and give the user the task verdict, timings and artifact paths
when `done` is true. Use the call history for a requested interaction trace.
