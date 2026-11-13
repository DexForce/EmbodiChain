# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Compare two finalized Newton models dumped by the platform scripts.

    python tools/compare_models.py <run_dir_a> <run_dir_b> [--out report.json]

Each run directory holds ``newton_model.npz`` (``--kind full`` runs of
``s01c_dexsim.py`` and ``s01c_embodichain.py``). Entity labels are compared
after removing the arena prefix (``arena_0/``). Arrays are compared element
by element; differing body/shape/joint/particle rows are named by label.
Exit code 0 = identical within tolerance, 1 = differences found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PREFIX = re.compile(r"^arena_\d+/")
# Index spaces of per-entity arrays, keyed by array-name prefix.
ENTITY = {"body_": "body_label", "shape_": "shape_label", "joint_": "joint_label"}
# Arrays that depend on allocation or bookkeeping rather than physics.
IGNORED = {
    "shape_label",
    "body_label",
    "joint_label",
    "articulation_label",
    "shape_source_ptr",
    "shape_source",
    "device",
}


def load(run_dir: Path) -> dict:
    return dict(np.load(run_dir / "newton_model.npz", allow_pickle=True))


def labels(model: dict, key: str) -> list[str]:
    return [PREFIX.sub("", str(v)) for v in model.get(key, [])]


def row_names(name: str, model: dict, rows) -> list[str]:
    for prefix, key in ENTITY.items():
        if name.startswith(prefix) and key in model:
            names = labels(model, key)
            if len(names) and max(rows) < len(names):
                return [names[r] for r in rows]
    return [str(r) for r in rows]


def compare(a: dict, b: dict, rtol: float, atol: float) -> dict:
    report = {
        "only_a": sorted(set(a) - set(b)),
        "only_b": sorted(set(b) - set(a)),
        "labels": {},
        "differences": {},
        "identical": 0,
    }
    for key in ENTITY.values():
        la, lb = labels(a, key), labels(b, key)
        if la != lb:
            report["labels"][key] = [
                (i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y
            ] + ([("count", len(la), len(lb))] if len(la) != len(lb) else [])
    for name in sorted(set(a) & set(b)):
        if name in IGNORED:
            continue
        x, y = a[name], b[name]
        if x.dtype.names is not None or y.dtype.names is not None:
            # Structs of device pointers (allocation, not physics).
            report.setdefault("skipped_struct_arrays", []).append(name)
            continue
        if x.shape != y.shape:
            report["differences"][name] = {"shape": [x.shape, y.shape]}
            continue
        if x.dtype.kind in "OUS":
            same = bool(np.all(x == y))
            if not same:
                report["differences"][name] = {"values": "text differs"}
            else:
                report["identical"] += 1
            continue
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        both_nan = np.isnan(xf) & np.isnan(yf)
        close = np.isclose(xf, yf, rtol=rtol, atol=atol) | both_nan
        if close.all():
            report["identical"] += 1
            continue
        bad = ~close
        rows = (
            np.unique(np.nonzero(bad.reshape(bad.shape[0], -1).any(1))[0])
            if x.ndim
            else np.array([0])
        )
        diff = np.abs(np.where(both_nan, 0.0, xf - yf))
        entry = {
            "max_abs_diff": float(np.nanmax(diff)),
            "count": int(bad.sum()),
            "size": int(x.size),
        }
        shown = rows[:8].tolist()
        if x.ndim:
            names = row_names(name, a, shown)
            entry["rows"] = [
                {
                    "row": r,
                    "label": n,
                    "a": np.asarray(x[r]).ravel()[:12].tolist(),
                    "b": np.asarray(y[r]).ravel()[:12].tolist(),
                }
                for r, n in zip(shown, names)
            ]
            entry["rows_total"] = int(len(rows))
        else:
            entry["a"], entry["b"] = x.tolist(), y.tolist()
        report["differences"][name] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(load(args.run_a), load(args.run_b), args.rtol, args.atol)
    report["run_a"], report["run_b"] = str(args.run_a), str(args.run_b)
    text = json.dumps(report, indent=1, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    print(
        f"identical arrays: {report['identical']}; differing: "
        f"{len(report['differences'])}; only in a: {report['only_a']}; "
        f"only in b: {report['only_b']}"
    )
    for key, value in report["labels"].items():
        print(f"label mismatch {key}: {value[:6]}")
    for name, entry in report["differences"].items():
        line = {k: v for k, v in entry.items() if k != "rows"}
        print(f"- {name}: {line}")
        for row in entry.get("rows", [])[:4]:
            print(f"    {row['label']}: a={row['a']} b={row['b']}")
    return 1 if report["differences"] or report["labels"] else 0


if __name__ == "__main__":
    sys.exit(main())
