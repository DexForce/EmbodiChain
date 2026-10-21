# Post-formal shutdown validation

The frozen study remains **29 qualified and one failed**. Its code and original
records are unchanged. The historical audit observed an existing process group
after a normal leader exit, but recorded no remaining PID identities; its
cause cannot be established retrospectively.

The current supervisor allows two seconds for normal helper exit and records
initial/final member identities. Any group remaining at the deadline, including
zombies or unknown membership, still fails. Live/unknown survivors undergo
termination and a further bounded observation. Waiting belongs to the stage's
clock. This change was applied only after frozen-source archiving.

Three controlled Linux process cases pass in [cpu/summary.json](cpu/summary.json):

- A helper exits normally after its leader, within the grace period.
- A persistent helper ignores SIGTERM, remains a strict failure, and is killed
  by SIGKILL. Its observed exit code is -9.
- An unreaped zombie remains a strict failure, without being described as an
  executing process or receiving ineffective signals.

[Final cleanup](cpu/cleanup-receipt.json) confirms all fixture groups are empty.
The executable [driver](qualify_group_exit.py) exercises the actual candidate
functions; it does not import the simulator or claim policy-training evidence.

A new dual-camera Offline-4 pipeline performs real generation, four updates
(16 chunks, 512 valid targets), four-worker standard SDK training, full-video
audit and two independent scenes. All stages and the outer supervisor exit 0,
with empty groups and no forced termination. Both scenes fail the physical task
criterion; this is engineering validation. See the
[physical receipt](physical-supervision/validation-summary.json).

The original failed checkpoint is evaluated separately on the same 20 fixed
development states: **0/20 successes**, 10/20 lifts, mean final error 0.193 m.
It exits normally with an empty group and no GPU placement violation. See the
[supplemental receipt](supplemental-evaluation/validation-summary.json).
The original eight control records retain their pre-validation hashes and
failed qualification. These additional scenes and their 286.7-second duration
are outside the formal cohort and all formal ratios.

Source paths and copied-byte hashes are in [retained-files.json](retained-files.json).
New diagnostic datasets/checkpoints stay at the paths in
[external-artifacts.json](external-artifacts.json); none is committed as a binary
corpus. No shadow summary, gate override, replacement seed or retrospective
qualification was used.
