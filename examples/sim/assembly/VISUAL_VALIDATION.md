# Visual assembly validation — 2026-10-25

These are recordings of actual simulated poses, with one physical environment,
a 1 ms timestep, pinned mjfit revision
`5833f465896e1e06bc4be4f003be847f5183a4db`, MuJoCo 3.12.0, Newton 1.6, and
DexSim 0.5.1rc1 plus the documented assembly SDK patches. Native execution used
an RTX 5090. The scripts retain the benchmark's geometry and physics; the
observer reads poses without setting them. Rendering takes place after the
physical rollout. Convex-piece colors and gear orientation witnesses affect
display only. Their timestamps remain physical simulation times.

The reference mechanics follow upstream
[drop_peg.py](https://github.com/younghyopark/mjfit/blob/5833f465896e1e06bc4be4f003be847f5183a4db/examples/drop_peg.py)
and [spin_nut.py](https://github.com/younghyopark/mjfit/blob/5833f465896e1e06bc4be4f003be847f5183a4db/examples/spin_nut.py).
The screw receives clockwise angular velocity only; its axial velocity evolves
through contact. Only the gear driver receives torque; both gears retain real
hinges and the follower receives no command.

| Case | Backend and route | Recorded behavior | Evidence scope |
|---|---|---|---|
| Circle, 100 µm clearance, 0.25 s | Native Newton/Warp, direct | Fall 15.000068 mm; insertion 10.000068 mm | Open aperture permits seating; one plate body owns all convex walls |
| Same circle, 0.25 s | Native Newton/Warp, USD | Same final drop and insertion | Both asset routes produce the same measured endpoint |
| Circle offset +20 mm in X, 0.25 s | Native Newton/Warp, direct | Fall 5.000075 mm; insertion only 0.000076 mm | Solid rim blocks the peg; classified as not inserted |
| M10, 1.5 s, 15 rad/s | Native Newton/Warp, direct | Advance 5.137766 mm vs theoretical 5.371479 mm; error 4.351% | Contact-driven screw advance meets this single probe's 5% behavior threshold |
| Same M10 | Native MuJoCo CPU reference | Advance 5.193374 mm; error 3.316% | Independent source reference |
| 18:24 gear pair, 1.5 s | Native MuJoCo CPU reference | Driver 12.760707 rad; ratio relative error 0.04951% | Reference contact-driven gearing |
| Same gear pair | Native Newton/Warp, direct | Failed: `mjSOL_PGS is unsupported.` | No native gear pass or movie claimed |

Raw artifacts are under `outputs/assembly_visual_verified/<case>/`: `asset.json`,
`trace.json`, `demo.gif` or `demo.mp4`, and `demo.frames.png`. The machine-readable
comparison is `outputs/assembly_visual_verified/measurements.json`. Its trace
files preserve source metadata and actual poses. The initial peg metrics were
recomputed from unchanged recorded poses after correcting a rim-stop false
positive; this is stated in their `scoring_note`. The regression covers seated,
rim-blocked and outside-plate endpoints, including scaled blind holes.

The live Viser scene was opened and visually checked. Its rigid geometry uses
category colors; the recorded view and inspection USD retain per-piece colors.
The viewer can hold the final scene for inspection, then exits after the
configured duration. This differs from a looping recording.

These probes establish visual and coarse physical behavior for the listed
cases. They do **not** qualify the complete migration. Native results remain
`unverified` because contact-overflow accounting is unavailable through the
current Scene facade; native teardown also emits SDK entity-balance warnings.
They do not establish the full µm clearance matrix, long-run stability, large
batch isolation/capacity, or trained blind-assembly policy performance. The
gear solver incompatibility remains an explicit integration gap.

Focused production tests: 27 passed across `test_assembly.py` and
`assembly/test_probes.py`. Black 26.10.1, the example launcher help path, and
the public API documentation gate pass. Context impact was reviewed; the
existing assembly ownership and qualification guidance remains accurate.

## Native DexSim window preview

Both verified behaviors were also run with an actual DexSim Hybrid window,
1280 x 800 output, 16 samples per pixel and OptiX denoising. Physics stayed at
1 kHz. Circle used 0.25 s simulation time and 0.1x playback; M10 used 1.5 s
simulation time and 0.4x playback. Both produced finite states and retained the
same source contact configuration.

| Case | Measured behavior | Render / collision evidence |
|---|---|---|
| Circle, 100 µm clearance | Fall 15.000068 mm; insertion 10.000068 mm | Complete plate and peg visible; hole retained |
| M10 | Advance 5.138384 mm vs ideal 5.371479 mm; error 4.3395% | Complete nut and bolt visible; actual Newton model has 186 nut and 339 bolt colliders, plus the floor |

Artifacts are `outputs/assembly_dexsim_window/final/circle_obj/` and
`outputs/assembly_dexsim_window/final/m10_obj/`. Their `native_window.mp4` and
GIF previews contain native DexSim RGB. The original window pixel-readback API
is unavailable in this SDK, so a native Camera with matching pose, projection
and clipping planes records the same scene. Its initial buffer warm-up does
not advance physics. Actual motion was checked from the initial and final RGB
frames, rather than accepting a nonblack frame containing only the nut.

The native window uses independent OBJ/MTL appearance because the installed
SDK's procedural multi-mesh construction left the bolt invisible. File loading
preserves the source triangle surfaces and material groups; physical Spawn
descriptors remain unchanged. An independent initialization certificate at
`final/physics_certificate/m10.json` confirms 526 actual Newton shapes including
the floor, with the same asset SHA256 as the recorded M10 rollout. This also
distinguishes a canonical aggregate collision-query mesh from the actual
backend collider count. The SDK procedural-render issue remains unresolved;
the native preview does not qualify that failing path or full physics health.

The private visual-export regression tests bring the focused assembly and
probe selection to **29 passed**. Public API coverage and formatting pass.

## Asset shading correction

The shared visual exporter now writes explicit OBJ corner normals and USD
`faceVarying` normals for source mesh faces. It retains the SDK primitive
normals for smooth cylinder sides and independent caps. This corrects the
rounded hex head, dented nut and corrugated plate caused by importer-generated
vertex averaging. Visual cache format 2 invalidates the previous OBJ assets.
No vertex positions, triangle topology, colliders, mass or inertia changed.

Native DexSim reruns at the same camera settings cover M10, the colored circle
drop and the default USB-C drop. Artifacts and before/after comparison are
under `outputs/assembly_render_fix/`; source asset hashes and camera settings
match the earlier previews. Circle falls 15.000068 mm, USB-C falls 11.200003 mm,
and M10 advances 5.133064 mm (4.4385% lead error). Native object visibility and
backend collider counts were checked before recording. The physics health
status remains `unverified` for the existing overflow-diagnostic limitation.

Focused regressions now report **31 passed**, including rotated hard-edge
normals, cylinder cap/side normals, source triangle preservation and USD normal
interpolation. USD geometry/mass acceptance remains unchanged for all three
assets. API coverage, Black 26.10.1 and context checks pass.

## Many-pegs and hinged gears — 2026-10-26

Official `many_pegs` now has a native example entry point. The 25-primitive,
seed-0 scene retains 50 bodies and 1050 body colliders. At 0.25 ms steps and
0.25 s, all 25 pegs seat. Its native Camera/window recording and full pose trace
are in `outputs/assembly_many_pegs/native_final/`; visibility and every backend
collider count are checked before recording. This is one scene with 25 cells,
not heterogeneous arena replication.

All three official gear pairs have real unlimited hinges, source passive
damping/friction and authored mass/inertia. Thirty base-profile cases plus six
explicit parameter cases pass the short-run diagnostic gates. Independent
full-tensor/COM/contact checks and six complete live-property captures also
pass. The final Warp CG native preview is under
`outputs/assembly_many_gears/native_cg_final/`; its cumulative ratio error is
0.02854% over 1.5 s, with 19 and 25 collision pieces retained. CPU PGS has a separate
recording and requires the optional isolated Newton patch.

Independent Warp reference initialization is explicit. The original
`cpu_compiled` protocol keeps near-zero CPU-derived translation inverse weights
and produces no follower rotation in three gear references. A controlled A/B
that only calls public `set_const_0` changes derived inverse weights while
retaining authored geometry, mass, inertia and contact parameters; all three
references then engage. Public `warp_recomputed` runs confirm all three passes,
with 0.05299%,0.08534%,0.02091% ratio errors. Old failures are retained. Effective
solver tolerance and MultiCCD flags are recorded; CPU and GPU solver profiles
are not pooled into a trajectory-equivalence claim.

The controlled 200 µm many-pegs reference matches Newton's GPU initialization,
MultiCCD-off and effective 1e-6 tolerance. Its maximum relative XYZ endpoint
delta is 0.784861 µm, with 25/25 within the 10 µm diagnostic. The original differing
protocol's 23.637 µm failure remains separate. At 20/50/100 µm, refined geometry
still exposes false planar classification or nonfinite state. A 32k candidate
capacity retry removes overflow but not the numerical failure. No collider is
deleted, thickened or hidden to obtain a pass.

Complete three-table reports and strict JSON data are
`outputs/assembly_many_pegs/ALIGNMENT_REPORT.md` and
`outputs/assembly_many_gears/ALIGNMENT_REPORT.md`. All conclusions are scoped
to these recorded experiments; complete quasistatic, large-batch, robot-control,
RL and real-device qualification remain outside this round.
