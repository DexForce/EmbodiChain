# Optional Newton CPU PGS compatibility patch

`newton_cpu_pgs.patch` targets the installed Newton **1.6.0** source file
`newton/_src/solvers/mujoco/solver_mujoco.py`, with unpatched SHA-256
`981568ab792dbc9e82392d57da3cbb507ef1c0d8c88e3b8b7cafe5065945c3cf`.
It is separate from the DexSim assembly patches and does not add PGS to
MuJoCo Warp. Apply it only to a private copy of this exact dependency;
`patch --dry-run` and the checksum check must succeed before use.

The installed Newton CPU backend also allocates MuJoCo Warp arrays as a
property staging store. That allocation rejects the PGS solver, although the
actual CPU dynamics would use native MuJoCo. The patch creates the staging
arrays from an independent model copy with supported metadata. The actual
CPU model retains PGS throughout; every dynamics step calls native
`mujoco.mj_step`. The Warp dynamics path remains unchanged and rejects PGS.
This retains staging arrays because the existing property update kernels
require them; it does not eliminate GPU memory use when the Newton model is
on CUDA.

Two additional corrections are necessary for source contact fidelity:

- Mirror friction, `solref`, `solimp`, `solmix`, margin and gap from the staging
  arrays into the CPU model after shape property updates.
- Disable the CPU midphase BVH. Conversion can change geometry frames after
  compilation, leaving that tree stale. In the observed gear model, exact
  `mj_geomDistance` reported 2.346 mm penetration while `mj_forward` generated
  zero contacts. Disabling `mjDSBL_MIDPHASE` restored eight contacts at the
  same pose. `mj_setConst` did not rebuild this tree. The exact narrowphase
  remains enabled; the cost is a CPU collision performance tradeoff.

This patch scopes validation to the passive assembly profile. It does not
claim that all CPU runtime property changes, arbitrary articulations,
contact sensors, or large heterogeneous batches have been qualified.

A safe application pattern is:

```bash
# Use the active environment's site-packages path, and a new empty directory.
python - <<'PY'
from pathlib import Path
import hashlib
import shutil
import tempfile
import newton
source = Path(newton.__file__).parent
relative = Path('_src/solvers/mujoco/solver_mujoco.py')
expected = '981568ab792dbc9e82392d57da3cbb507ef1c0d8c88e3b8b7cafe5065945c3cf'
assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected
root = Path(tempfile.mkdtemp(prefix='embodichain-newton-cpu-pgs-'))
shutil.copytree(source, root / 'newton')
print(root)
PY
patch --dry-run -p1 -d /absolute/path/to/the/new/copy < scripts/patches/newton_cpu_pgs.patch
patch -p1 -d /absolute/path/to/the/new/copy < scripts/patches/newton_cpu_pgs.patch
```

Prepend that directory to the runtime's existing `PYTHONPATH`; keep the
DexSim assembly patches in their separate overlay. Select the profile
explicitly through the EmbodiChain physics configuration:

```python
solver_cfg = {
    'solver_type': 'mujoco_warp',
    'use_mujoco_cpu': True,
    'solver': 'pgs',
    'use_mujoco_contacts': True,
    'separate_worlds': False,
    'iterations': 100,
    'update_data_interval': 1,
}
```

The `solver_type` is the DexSim configuration class name. Report the actual
executor as **EmbodiChain Newton / native MuJoCo CPU PGS**, rather than Warp.
The CPU path still writes Newton state in FP32, so CPU execution alone does
not make the integrated dynamics FP64 equivalent to independent MuJoCo.

CPU-only smoke measurements from the isolated dependency copy:

- A free 1 kg box after 100 steps at 1 ms differed from independent MuJoCo
  CPU by 3.99e-8 m in height, with zero warnings. Authored `solref`, `solimp`
  and friction were retained.
- The source 18:24 gear pair imported through Newton's public MJCF parser
  retained true hinges and contact-only coupling. At 1.5 s with 0.02 N m
  driver torque, driver/follower angles were 13.229881 / -9.914173 rad,
  ratio error 0.08303%, zero warnings and peak contact count two. This is a
  dependency smoke, separate from EmbodiChain's URDF and USD integration
  acceptance.

The immutable source benchmarks in
`scripts/benchmark/assembly/gear_reference.py` run independently of this
patch. Always retain source PGS, source CG and integrated profiles as
separate evidence; switching to CG must be explicit and measured.
