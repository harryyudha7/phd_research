"""Mechanically assemble the three A/B notebooks from vetted mesh-solver cells.

The copied cells are the mesh/CG/target blocks.  All reconstruction cells import
``fracture_hardcurl_common`` and are intentionally identical across variants.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS = [
    {
        "key": "conforming_tri",
        "title": "Conforming triangular mesh",
        "source": "LCG_Deng_flux_reconstruction_MMS_conforming_P0.ipynb",
        "output": "LCG_fracture_MMS_hardcurl_PINN_conforming_tri_AB.ipynb",
        "cells": [1, 2, 3, 4, 5],
    },
    {
        "key": "nonconforming_tri",
        "title": "Nonconforming opposite-diagonal triangular mesh",
        "source": "LCG_Deng_flux_reconstruction_MMS_nonconforming_opposite_diagonal_P0.ipynb",
        "output": "LCG_fracture_MMS_hardcurl_PINN_nonconforming_tri_AB.ipynb",
        "cells": [1, 2, 3, 4, 5],
    },
    {
        "key": "nonconforming_rect",
        "title": "Nonconforming rectangular mesh",
        "source": "LCG_Deng_flux_reconstruction_MMS_rect_nonconforming_P0.ipynb",
        "output": "LCG_fracture_MMS_hardcurl_PINN_nonconforming_rect_AB.ipynb",
        "cells": [1, 2, 3, 4, 5, 6, 7],
    },
]


TOP = r"""# Fracture MMS hard-curl PINN — {title} — Options A and B

This notebook reconstructs the bulk Darcy flux for the single-fracture MMS on
\(\Gamma=\{{(t,t):0\le t\le1\}}\). The mesh-specific block below is ported from
`{source}`. The reconstruction itself is imported from
`fracture_hardcurl_common.py` and is mesh-independent: it sees points and oriented
dual-face segments, never elements.

## Fixed conventions

- The fracture is oriented from \(A=(0,0)\) to \(B=(1,1)\), with
  \(\tau=(1,1)/\sqrt2\) and one fixed normal
  \(n_\Gamma=(-1,1)/\sqrt2\), pointing to \(\Omega^+=\{{y>x\}}\).
- Everywhere,
  \([[q\cdot n_\Gamma]]=q^+\cdot n_\Gamma-q^-\cdot n_\Gamma=\lambda\).
- The bulk balance audited on every dual CV is
  \(\oint_{\partial\omega}q\cdot n\,ds-\int_\omega f_m\,dx
  -\int_{\Gamma\cap\omega}\lambda\,ds=0\).
- Both exact-\(\lambda\) and discrete-\(\lambda_h\) right-hand sides are reported.
- `float64` and fixed seeds are used throughout. Every construction gate prints its
  tolerance and raises immediately on failure.
- The `source` CV class is retained in the common audit schema but is empty/N/A for
  this smooth distributed MMS source; \(\int_\omega f_m\) is still evaluated on
  every CV.
- Because the fracture is corner-to-corner, there is no artificial interface
  extension. Option B reports its extension-continuity penalty explicitly as N/A.

The conforming solve uses the established independent coarse P0 multiplier mesh
with \(h_\lambda=2h\); this is the multiplier used by \(q_{p,\lambda}\).
"""


def markdown(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build(spec: dict) -> None:
    src = json.loads((HERE / spec["source"]).read_text())
    top = TOP.replace("{title}", spec["title"]).replace("{source}", spec["source"])
    top = top.replace("{{", "{").replace("}}", "}")
    cells = [markdown(top)]
    cells.append(markdown("## 1. Mesh-specific MMS and CG–LMDFM block\n\nPorted without modifying the source notebook."))
    for idx in spec["cells"]:
        cells.append(code("".join(src["cells"][idx].get("source", []))))

    cells.extend([
        markdown("## 2. Shared hard-curl controls\n\nThese controls are identical in all three notebooks."),
        code(f"""import importlib
import sys
_hardcurl_dir = pathlib.Path.cwd() / "fenicsx/code/fracture problem"
if not (_hardcurl_dir / "fracture_hardcurl_common.py").exists():
    _hardcurl_dir = pathlib.Path.cwd()
if str(_hardcurl_dir) not in sys.path:
    sys.path.insert(0, str(_hardcurl_dir))
import fracture_hardcurl_common as hardcurl
importlib.reload(hardcurl)

VARIANT = {spec['key']!r}
HARD_REF = REF_DEMO
FACE_GAUSS_ORDER = 32
SOURCE_GAUSS_ORDER = 20
LINE_GAUSS_ORDER = 12
ADAM_STEPS_A = 2000
LBFGS_STEPS_A = 250
ADAM_STEPS_B = 2000
LBFGS_STEPS_B = 150
B_PENALTY_WEIGHTS = (1.0, 10.0, 100.0)

assert hardcurl.DTYPE == np.float64
hardcurl.set_fixed_seeds()
print("float64 convention: PASS")
print("Option B extension-continuity term: N/A (no artificial extension)")"""),
        markdown("## 3. Gate A0 — isolated line-source jump\n\nThis gate is mesh-free and network-free."),
        code("gate_A0 = hardcurl.gate_a0()"),
        markdown(r"## 4. Common dual-CV data and target fluxes" "\n\n" r"Dual faces crossing the fracture are split once. The split is used for one-sided CG targets and for the discontinuous line-source contribution; the smooth analytic \(q_{p,f}\) itself needs no fracture split."),
        code("""problem = hardcurl.build_problem(
    globals(), VARIANT, ref=HARD_REF,
    face_order=FACE_GAUSS_ORDER,
    source_order=SOURCE_GAUSS_ORDER,
    line_order=LINE_GAUSS_ORDER,
)"""),
        markdown(r"## 5. Gate A1 — conservation identity before training" "\n\n" r"The gate tests \(\psi=0\) and a randomly initialized network with matching exact/discrete line-source densities. Every nonempty CV class must satisfy `max|R_xi| <= 1e-13`."),
        code("""gate_A1 = hardcurl.gate_a1(problem)
fracture_1d_gate = hardcurl.fracture_1d_conservation_check(problem)"""),
        markdown(r"# Option A — line source in the particular field" "\n\n" r"\(q_A=q_{p,f}+q_{p,\lambda_h}+\nabla^\perp\psi\). The single smooth network has no interface penalty. The analytic trace jump of \(q_{p,\lambda_h}\) is exactly \(\lambda_h\), while the curl term is continuous."),
        markdown("## A2. Train the single stream-function network"),
        code("""option_A = hardcurl.run_option_a(
    problem, adam_steps=ADAM_STEPS_A, lbfgs_steps=LBFGS_STEPS_A,
    width=48, depth=3, lr=2.0e-3,
)"""),
        markdown(r"## A3. Verification battery" "\n\n" r"Option A is completed and audited before Option B begins. With the \(\lambda_h\)-consistent RHS, conservation should be at machine precision in every class. With exact \(\lambda\), the cut-CV residual is the discrete multiplier consistency error."),
        code("""hardcurl.plot_option_a_verification(problem, option_A)
option_A_audit = option_A["audit"]
print("Option A completed: all construction gates passed before Option B.")"""),
        markdown(r"# Option B — two subdomain stream functions" "\n\n" r"\(q_B^\pm=q_{p,f}+\nabla^\perp\psi_\pm\). No \(q_{p,\lambda}\) is used. The multiplier is imposed only through the difference constraint \((q_B^+-q_B^-)\cdot n_\Gamma=\lambda_h\); it is never split 50/50. Each cut data face contributes its one-sided segments. The corner-to-corner geometry has no artificial extension, so that term is reported as N/A."),
        markdown("## B1–B2. Penalty sweep and training"),
        code("""option_B = hardcurl.run_option_b(
    problem, penalty_weights=B_PENALTY_WEIGHTS,
    adam_steps=ADAM_STEPS_B, lbfgs_steps=LBFGS_STEPS_B,
    width=24, depth=2, lr=2.0e-3,
)"""),
        markdown("## B3. Verification battery and penalty floor"),
        code("hardcurl.plot_verification(problem, option_A, option_B)"),
        markdown("## Final A/B comparison\n\nThe table reports measured accuracy, conservation, parameter count, and wall time without ranking beyond the numerical evidence."),
        code("comparison_rows = hardcurl.comparison_table(problem, option_A, option_B)"),
        code("""import json as _json
SUMMARY_DIR = pathlib.Path("result_fracture_MMS_hardcurl_AB")
SUMMARY_DIR.mkdir(exist_ok=True)
mesh_summary = {
    "mesh": VARIANT,
    "A_flux_error": comparison_rows[0]["face_flux_RMSE_exact"],
    "B_flux_error": comparison_rows[1]["face_flux_RMSE_exact"],
    "A_worst_Rxi": comparison_rows[0]["worst_Rxi"],
    "B_worst_Rxi": comparison_rows[1]["worst_Rxi"],
    "A_jump_error": comparison_rows[0]["jump_RMSE_lambda_h"],
    "B_jump_error": comparison_rows[1]["jump_RMSE_lambda_h"],
}
summary_file = SUMMARY_DIR / f"{VARIANT}_summary.json"
summary_file.write_text(_json.dumps(mesh_summary, indent=2) + "\\n")
print("Saved", summary_file)"""),
        markdown("### Conclusion\n\nOption A is conservative by construction with the discrete multiplier-consistent RHS: its line-source particular field carries the exchange distribution and its curl correction cannot change any closed-CV balance. Option B reaches only the penalty-controlled conservation floor shown above; its two smaller networks trade exact jump enforcement for a soft interface constraint. The measured flux errors, parameter counts, and wall times are reported in the table."),
    ])

    if spec["key"] == "nonconforming_rect":
        cells.extend([
            markdown("## Cross-mesh summary\n\nRun the first two notebooks and save/pass their `comparison_rows` when assembling a publication table. The schema below is fixed across meshes and demonstrates that the reconstruction code and audit columns are unchanged by fracture alignment."),
            code("""cross_mesh_rows = []
for _mesh in ("conforming_tri", "nonconforming_tri", "nonconforming_rect"):
    _path = SUMMARY_DIR / f"{_mesh}_summary.json"
    if _path.exists():
        cross_mesh_rows.append(_json.loads(_path.read_text()))
    else:
        print(f"N/A: run the {_mesh} notebook to create {_path.name}")

print("mesh                  A flux RMSE    B flux RMSE    A worst R_xi   B worst R_xi")
for row in cross_mesh_rows:
    print(f"{row['mesh']:<22} {row['A_flux_error']:12.4e} {row['B_flux_error']:12.4e} "
          f"{row['A_worst_Rxi']:13.4e} {row['B_worst_Rxi']:13.4e}")"""),
        ])

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / spec["output"]).write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", spec["output"], "cells=", len(cells))


if __name__ == "__main__":
    for item in VARIANTS:
        build(item)
