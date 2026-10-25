# Many-pegs integration and precision alignment

Geometry conversion passes for all nine prepared profiles. Full 20–200 µm GPU precision qualification remains incomplete: 20 µm is blocked or nonfinite, and 50/100 µm remain nonfinite. The more controlled 200 µm comparison, with matching physical options and Warp constant initialization, achieves a maximum endpoint delta of **0.785 µm**, below the 10 µm diagnostic; native health remains unverified.

The official public generation path is preserved: `sample_pegs(25, seed=0, family="primitive")`, `build(..., standalone=False)`, `grid(..., spacing=0.3)`, compile and serialize. Every scene has 50 physical bodies, 25 grid cells and one arena. The generator revision is `5833f465896e1e06bc4be4f003be847f5183a4db`. Public shape settings in precise profiles set `max_error=0.1×clearance`; sampled and actual configs are both recorded.

All selected CPU references and micro GPU runs use 0.25 s, 0.25 ms steps and 100 solver iterations. Micro GPU capacity is 16384 contacts / 98304 constraint rows, with 512 MiB host extraction arena. Runtime versions: MuJoCo 3.12.0, MuJoCo Warp 3.12.0, Newton 1.6.0, Warp 1.17.0 and DexSim 0.5.1rc1 with the documented source-contact patch. Raw data, exact hashes, per-cell metrics and settings are in `many_pegs.json`.

## Time & Memory

These are single correctness-run wall observations. Different instrumentation and first-step compilation are included; this is not a throughput or latency ranking. Memory is unmeasured.

| Execution scope | cost_time_ms per step, observed range | timed/planned cases | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |
| --- | ---: | ---: | --- | --- | --- |
| MuJoCo CPU, all 9 preparation profiles | 1.645–12.350 | 9/9 | unmeasured | unmeasured | unmeasured |
| Native MuJoCo Warp, precise 4 profiles | 3.616–5.043 | 4/4 | unmeasured | unmeasured | unmeasured |
| EmbodiChain Newton, precise 4 profiles | 4.174–5.524 | 3/4 | unmeasured | unmeasured | unmeasured |
| EmbodiChain Newton, official profile direct/USD convergence | 4.785–4.960 | 4/4 | unmeasured | unmeasured | unmeasured |
| Diagnostic native Warp 20 µm, capacity 32768 | 5.636 | 1/1 | unmeasured | unmeasured | unmeasured |
| Native Warp official 500 µm reference | 8.944 | 1/1 | unmeasured | unmeasured | unmeasured |
| Matched-parameter / Warp-recomputed precise references | 3.428–4.782 | 4/4 | unmeasured | unmeasured | unmeasured |

## Success & Other Metrics

Geometry and CPU behavior are separate columns. USD checks include exact parent ownership, every collision piece and authored mass/COM/inertia. Every row passes the 1 µm geometry and 1e-5 relative mass/inertia thresholds. Independent Shapely union of compiled prism slices measures actual polygon gap, including islands; the largest gap error across all profiles is 0.0997245 µm, with zero polygon overlap. This is not an oracle for intended curved shapes or quasistatic dynamics.

| Profile / clearance | bodies / colliders | USD geometry error µm | compiled gap error µm | CPU seated / 25 | CPU max axial center error µm | micro GPU result (Warp; EmbodiChain) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| official / 500 µm | 50 / 1050 | 0.0196 | 0.0833 | 25/25 | 3.303 | reference passed 25/25; native direct/USD unverified |
| unchanged tessellation / 20 µm | 50 / 1082 | 0.0218 | 0.0917 | 25/25 | 0.526 | not run in this profile |
| unchanged tessellation / 50 µm | 50 / 1088 | 0.0201 | 0.0997 | 25/25 | 0.520 | not run in this profile |
| unchanged tessellation / 100 µm | 50 / 1085 | 0.0215 | 0.0778 | 25/25 | 1.645 | not run in this profile |
| unchanged tessellation / 200 µm | 50 / 1096 | 0.0193 | 0.0882 | 25/25 | 0.895 | not run in this profile |
| precise / 20 µm | 50 / 4813 | 0.0218 | 0.0919 | 25/25 | 2.946 | failed: nonfinite+overflow; failed: planar classifier |
| precise / 50 µm | 50 / 3251 | 0.0201 | 0.0821 | 25/25 | 0.555 | failed: nonfinite; failed: nonfinite |
| precise / 100 µm | 50 / 2426 | 0.0215 | 0.0884 | 25/25 | 3.478 | failed: nonfinite; failed: nonfinite |
| precise / 200 µm | 50 / 1882 | 0.0193 | 0.0828 | 25/25 | 4.237 | passed endpoint/health; unverified endpoint25/25 |

The unchanged-tessellation 20/50 µm profiles retain the official approximately 100 µm curve chord setting and emit 28/29 upstream warnings. Precise profiles remove these warnings by tightening public curved shape specs, without deleting any collision piece. Changing the profile changes source geometry and authored mass; each native/USD comparison retains its own source mass rather than forcing different profiles to share mass.

## Default-profile endpoint convergence

At 500 µm clearance, the matched CPU/Newton maximum plate-relative XYZ endpoint difference is 162.109 µm for 0.5 ms steps and 22.624 µm (direct) / 23.584 µm (USD) for 0.25 ms steps. Both 0.5 ms routes have 23/25 endpoints within the displayed 50 µm and 25 µm diagnostics; both 0.25 ms routes have 25/25. Direct-versus-USD dynamic endpoint deltas are 0.240 µm and 1.019 µm respectively. These dynamic deltas do not replace the independent static geometry measurement. All four native runs retain 50 bodies and 1050 actual collision parts. Detailed data are in `endpoint_comparison.json`.

The independent official-profile Warp 3.12 reference now passes: 25/25 seated, finite state and zero overflow flags at 0.25 ms, 0.25 s and capacity 8192. Newton direct/USD maximum plate-relative XYZ endpoint deltas against that reference are **2.587 µm / 1.800 µm**, with all 25 cells within the displayed 25 µm diagnostic. Raw reference and per-cell comparisons are in `official_warp_reference/cell_0.json` and `official_warp_reference/endpoint_comparison.json`. Native facade health remains unverified.

**Solver-profile difference:** the reference records MultiCCD on (`disableflags=0`, effective tolerance about 1e-6), while the older Newton convergence runs inherit Newton's `enable_multiccd=False` default. The original source MuJoCo option flags are zero. Same version and geometry do not isolate integration error when this contact-generation option differs. The original default-profile and 200 µm comparisons use these different recorded route profiles; their deltas cannot be attributed solely to integration. The controlled reference series below explicitly matches physical options and the Warp constant initialization stage, while native health remains unverified.

## Precise 200 µm same-version endpoint comparison

With the original CPU-initialized / MultiCCD-on reference, all three backends produce finite, seated 25/25 endpoints at this profile. EmbodiChain versus native Warp 3.12 has a maximum plate-relative XYZ endpoint delta of23.637 µm at cell 11; 24/25 cells meet `max(2µm,0.05×200µm)=10µm`. The endpoint added-error diagnostic therefore fails. CPU versus Warp differs by265.090 µm and CPU versus EmbodiChain by266.490 µm, also dominated by cell 11 lateral drift. Free-drop endpoints are not a quasistatic clearance oracle, so no quasistatic gate is marked passed or failed from these values.

The native 200 µm case retains all 50 bodies and 1882 collision parts. Native overflow/cross-cell accounting remains unavailable through the Scene facade, so its status stays unverified. The independent Warp 200 µm run reports finite states and zero overflow flags; this passes only the stated passive endpoint/health probe.

A separate controlled reference series aligns Newton solver, effective tolerance 1e-6, MultiCCD off and `warp_recomputed` constants using public `set_const_0`, matching the Newton initialization stage. Capacity is 32768 contacts / 196608 constraints. The 20/50/100 µm source Warp references still fail with 225 nonfinite generalized-state entries each and zero overflow. The 200 µm source reference passes all 25 endpoints with zero overflow and maximum axial center error about 0.111 µm.

Against the original Newton 200 µm result, this controlled reference yields **0.784861 µm maximum plate-relative XYZ endpoint delta** at cell 11; all 25 cells meet the 10 µm endpoint diagnostic. Maximum axial delta is 0.003717 µm. The earlier 23.637 µm CPU-initialized / MultiCCD-on reference result remains recorded as a different-protocol diagnostic failure. This comparison controls more physical and initialization settings, while allocation capacity still differs (source Warp 32768, original Newton 16384) and native overflow health is unavailable. It qualifies the endpoint diagnostic only. Data are in `source_warp_recomputed/cell_0.json` through `cell_3.json` and `source_warp_recomputed/endpoint_comparison.json`.

## Failure evidence and capacity

For precise 20 µm, native Newton rejects `p21_peg_ring_296` as planar. The unchanged source collider has 6 vertices, 8 triangles and nonzero volume 1.05677e-12 m³, and compiles independently in MuJoCo 3.12. Newton uses a 19.364 nm planarity tolerance while the selected-plane distance is 17.497 nm. Its classifier treats the thin three-dimensional prism as planar. Source vertices already have float32 values; another local float32 conversion adds zero coordinate error. `planarity_analysis.json` contains exact vertices, F32/F64 diagnostics and source locations. No piece was removed, thickened or substituted.

At capacity 16384, native Warp 20 µm produces nonfinite generalized state and a broadphase overflow flag. The log requests up to 23340 candidate slots. This demonstrates insufficient candidate capacity; it does not by itself isolate the NaN cause. At 50/100 µm, native Warp and EmbodiChain produce nonfinite state; Warp overflow counts are zero there. The common NaN cause is not established by this experiment. Native 50/100 µm binding counts still retain all 3251/2426 source pieces. Nonfinite values are stored as JSON nulls with explicit failed status.

The precise 20 µm CPU source needs 3889 actual contacts, 23334 constraint rows and 140004 sparse Jacobian entries at peak. These counts do not bound Warp broadphase candidate storage. The original Warp 20 µm log contains 107 broadphase-overflow warnings requesting 22289–23340 candidate capacity, above the configured 16384. The broadphase kernel stops storing candidate pairs beyond its capacity, so this run is invalid. CPU actual-contact counts and the nominal 4.21× contact/constraint margin do not establish GPU buffer sufficiency. CPU forward arena peak use is 8.86 MiB. Warp readback rebuilds a dense CPU Jacobian despite sparse simulation; worst configured storage is 112.5 MiB, exceeding the original22 MiB arena. Raising only host memory to 512 MiB preserves all checked model geometry and masses, and fixes that readback allocation. It does not fix NaNs or the planar classifier. No claim of GPU out-of-memory failure is made.

The separate source Warp 20 µm capacity follow-up is complete at `nconmax=32768`, `njmax=196608`, preserving geometry, mass and contact parameters. It reports **zero overflow flags and no overflow warnings, but the same 225 nonfinite generalized-state entries**, so the run still fails. Increasing candidate capacity removed the observed overflow and did not resolve the numeric failure. The 1.404× margin over the original largest logged demand is not a proven future bound. The original 16384-capacity failure is preserved; the common-profile geometry/results table and leaderboard retain that protocol. Retry data are in `precise_20um_capacity32768/cell_0.json` and its log.

## Leaderboard

Common scope: exactly the four precise profiles, 100 planned pegs per backend. Ranks describe coarse endpoint seating only. Construction failures and nonfinite runs count as failed planned pegs; these ranks do not establish precision, stability or learning performance.

| Rank | execution backend | overall_success_rate, coarse seating | seated/planned pegs | finite/planned cases | physical precision status |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | MuJoCo CPU | 100% | 100/100 | 4/4 | CPU passive reference only |
| 2 | native Warp, original protocol | 25% | 25/100 | 1/4 | 3 micro profiles failed; 1 passive probe passed |
| 3 | native Warp, matched / recomputed | 25% | 25/100 | 1/4 | 3 micro profiles failed; 1 passive probe passed |
| 4 | EmbodiChain Newton / Warp | 25% | 25/100 | 1/4 | 3 micro profiles failed; 1 remains unverified |

All runs use one deterministic seed and one scene; grid cells are not isolated arenas. Multi-seed distributions, quasistatic/contact-onset probes, long-time stability, Default backend and large arena counts remain unqualified. No performance claim follows from the displayed timings. The combined project report owns broader hardware and demo evidence.
