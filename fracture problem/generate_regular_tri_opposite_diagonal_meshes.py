"""Generate regular triangular unit-square meshes with the opposite diagonal.

The physical MMS fracture used by the notebooks is the global diagonal
``(0, 0) -> (1, 1)``.  These bulk meshes deliberately split each square with
the other local diagonal, ``(i+1, j) -> (i, j+1)``, so that the MMS fracture is
nonconforming.

The script is idempotent: existing ``.msh`` files are left untouched.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import gmsh


def generate_mesh(ref: int, outdir: pathlib.Path, overwrite: bool = False) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)
    filename = outdir / f"regular_tri_opposite_diagonal_{ref}.msh"
    if filename.exists() and not overwrite:
        print(f"skip existing: {filename}")
        return filename

    n = 2**ref
    h = 1.0 / n

    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.model.add(f"regular_tri_opposite_diagonal_{ref}")

    try:
        points = [[None for _ in range(n + 1)] for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                points[i][j] = gmsh.model.occ.addPoint(i * h, j * h, 0.0)

        triangle_surfaces: list[int] = []
        for i in range(n):
            for j in range(n):
                p00 = points[i][j]
                p10 = points[i + 1][j]
                p11 = points[i + 1][j + 1]
                p01 = points[i][j + 1]

                # Lower-left triangle: p00 -> p10 -> p01.
                l1 = gmsh.model.occ.addLine(p00, p10)
                l2 = gmsh.model.occ.addLine(p10, p01)
                l3 = gmsh.model.occ.addLine(p01, p00)
                loop1 = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                triangle_surfaces.append(gmsh.model.occ.addPlaneSurface([loop1]))

                # Upper-right triangle: p10 -> p11 -> p01.
                l4 = gmsh.model.occ.addLine(p10, p11)
                l5 = gmsh.model.occ.addLine(p11, p01)
                l6 = gmsh.model.occ.addLine(p01, p10)
                loop2 = gmsh.model.occ.addCurveLoop([l4, l5, l6])
                triangle_surfaces.append(gmsh.model.occ.addPlaneSurface([loop2]))

        gmsh.model.occ.synchronize()

        omega = [tag for dim, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, omega, 1)
        gmsh.model.setPhysicalName(2, 1, "Omega")

        # Each triangle owns its own curve entities, so counting curve tags would
        # label duplicated internal curves as boundary.  Use geometry instead.
        boundary_curves: list[int] = []
        tol = 1.0e-12
        for dim, tag in gmsh.model.getEntities(1):
            xmin, ymin, _zmin, xmax, ymax, _zmax = gmsh.model.getBoundingBox(dim, tag)
            on_outer_boundary = (
                abs(xmin) < tol and abs(xmax) < tol
                or abs(xmin - 1.0) < tol and abs(xmax - 1.0) < tol
                or abs(ymin) < tol and abs(ymax) < tol
                or abs(ymin - 1.0) < tol and abs(ymax - 1.0) < tol
            )
            if on_outer_boundary:
                boundary_curves.append(tag)
        if boundary_curves:
            gmsh.model.addPhysicalGroup(1, boundary_curves, 3)
            gmsh.model.setPhysicalName(1, 3, "Boundary")

        for dim, tag in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(tag, 2)
        for surface in triangle_surfaces:
            gmsh.model.mesh.setTransfiniteSurface(surface)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        gmsh.write(str(filename))
        print(f"wrote: {filename}")
    finally:
        gmsh.finalize()

    return filename


def parse_refs(values: list[str]) -> list[int]:
    refs: list[int] = []
    for value in values:
        if ":" in value:
            a, b = value.split(":", 1)
            refs.extend(range(int(a), int(b) + 1))
        else:
            refs.append(int(value))
    return sorted(set(refs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "refs",
        nargs="*",
        default=["1:7"],
        help="Refinement levels, e.g. 3 4 5 or 1:7. Default: 1:7.",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for ref in parse_refs(args.refs):
        generate_mesh(ref, args.outdir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
