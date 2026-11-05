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
"""Validated static tetrahedron specialization for scoped DexUni diagnostics."""

from __future__ import annotations

import ast
import numpy as np

__all__ = ["specialize_uniform_materials"]


def specialize_uniform_materials(source: str, materials: np.ndarray) -> str:
    """Embed immutable, uniform active material values without changing formulas.

    Args:
        source: Original kernel with zero surface loops already stripped.
        materials: Float32 tetrahedron rows (mu, lambda, damping).

    Returns:
        Equivalent source with one active-material branch removed.

    Raises:
        ValueError: Nonuniform, inactive, nonfinite or unsupported material layout.
    """
    materials = np.asarray(materials)
    if (
        materials.ndim != 2
        or materials.shape[0] == 0
        or materials.shape[1] != 3
        or materials.dtype != np.float32
        or not np.isfinite(materials).all()
        or not np.all(materials == materials[0])
        or not (materials[0, 0] > 0 or materials[0, 1] > 0)
    ):
        raise ValueError(
            "Uniform specialization requires immutable, finite, active float32 materials"
        )
    values = materials[0].tolist()
    branches = 0

    class Constants(ast.NodeTransformer):
        def visit_If(self, node: ast.If) -> ast.AST:
            nonlocal branches
            if any(
                isinstance(x, ast.Name) and x.id == "tet_materials"
                for x in ast.walk(node.test)
            ):
                node.test = ast.parse("wp.static(True)", mode="eval").body
                branches += 1
            return self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "tet_materials"
                and isinstance(node.slice, ast.Tuple)
            ):
                column = node.slice.elts[1]
                if isinstance(column, ast.Constant):
                    return ast.copy_location(
                        ast.parse(
                            f"wp.float32({values[column.value]!r})", mode="eval"
                        ).body,
                        node,
                    )
            return self.generic_visit(node)

    tree = Constants().visit(ast.parse(source))
    if branches != 1:
        raise ValueError(
            "Expected the original single active-tetrahedron material branch"
        )
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
