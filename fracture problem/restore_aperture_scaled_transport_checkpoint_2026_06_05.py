#!/usr/bin/env python3
"""Restore the aperture-scaled fracture-transport checkpoint from 2026-06-05.

This rolls back only the source edits made after the aperture-scaled run:
fracture Q/tip/storage scaling, TRANSPORT_CLIP, the simple fracture-balance
diagnostic label, and the looser MRST NNC mapping source block.

Usage:
  python3 restore_aperture_scaled_transport_checkpoint_2026_06_05.py
  python3 restore_aperture_scaled_transport_checkpoint_2026_06_05.py /path/to/notebook.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


NOTEBOOK_NAME = "LCG_nonconforming_PINN_flux_reconstruction_KoppelMartin_case1_2d.ipynb"


def replace_checked(text: str, old: str, new: str, label: str, expected: int = 1) -> str:
    count = text.count(old)
    if count == expected:
        return text.replace(old, new)
    if count == 0 and text.count(new) >= expected:
        print(f"{label}: already at checkpoint value")
        return text
    raise RuntimeError(f"{label}: expected {expected} match(es), found {count}")


def main() -> int:
    if len(sys.argv) > 1:
        notebook = Path(sys.argv[1])
    else:
        notebook = Path(__file__).with_name(NOTEBOOK_NAME)

    text = notebook.read_text()

    replacements = [
        (
            '    "TRANSPORT_CLIP = False  # keep off as a balance guard; clip dM diagnostics remain visible if enabled\\n",\n',
            '    "TRANSPORT_CLIP = True  # clip S to [0,1] each step; off to expose non-conservative overshoot\\n",\n',
            "TRANSPORT_CLIP",
            1,
        ),
        (
            '    "    q_elem = -K_F_VALUE * (p_right - p_left) / length\\n",\n',
            '    "    q_elem = -TRANSPORT_FRAC_APERTURE * K_F_VALUE * (p_right - p_left) / length\\n",\n',
            "LCG fracture Q scaling",
            1,
        ),
        (
            '    "        \\"pore_volume\\": TRANSPORT_FRAC_POROSITY * length,\\n",\n',
            '    "        \\"pore_volume\\": TRANSPORT_FRAC_POROSITY * TRANSPORT_FRAC_APERTURE * length,\\n",\n',
            "fracture pore volume scaling",
            2,
        ),
        (
            '    "        \\"aperture_convention\\": \\"aperture-free flux and pore volume\\",\\n",\n',
            '    "        \\"aperture_convention\\": \\"aperture-scaled flux and pore volume\\",\\n",\n',
            "aperture convention label",
            2,
        ),
        (
            '    "        q_face[1:-1] = -Kgamma * (p[1:] - p[:-1]) / (s[1:] - s[:-1])\\n",\n',
            '    "        q_face[1:-1] = -TRANSPORT_FRAC_APERTURE * Kgamma * (p[1:] - p[:-1]) / (s[1:] - s[:-1])\\n",\n',
            "MRST interior fracture Q scaling",
            1,
        ),
        (
            '    "        q_tip_sign = expected_dir if expected_dir != 0.0 else -1.0\\n",\n'
            '    "        q_face[0] = q_tip_sign * abs(float(tip_flux[a_idx]))\\n",\n'
            '    "        q_face[-1] = q_tip_sign * abs(float(tip_flux[b_idx]))\\n",\n'
            '    "        tip_A_flux = q_face[0]\\n",\n',
            '    "        q_tip_sign = expected_dir if expected_dir != 0.0 else -1.0\\n",\n'
            '    "        # MRST README reports tip_flux ~= K_Gamma dp/L, so scale it to volumetric flux.\\n",\n'
            '    "        q_face[0] = q_tip_sign * TRANSPORT_FRAC_APERTURE * abs(float(tip_flux[a_idx]))\\n",\n'
            '    "        q_face[-1] = q_tip_sign * TRANSPORT_FRAC_APERTURE * abs(float(tip_flux[b_idx]))\\n",\n'
            '    "        tip_A_flux = q_face[0]\\n",\n',
            "MRST tip fracture Q scaling",
            1,
        ),
        (
            '    "def fracture_balance_display_label(label, fracture):\\n",\n'
            '    "    if fracture.get(\\"kind\\") == \\"lcg\\":\\n",\n'
            '    "        return f\\"{label} (LCG q_gamma)\\"\\n",\n'
            '    "    return str(label)\\n",\n'
            '    "\\n",\n'
            '    "\\n",\n'
            '    "def print_fracture_single_phase_balance(label, fracture, exchange, gate_tol=None):\\n",\n'
            '    "    residual = fracture_single_phase_residual(fracture, exchange)\\n",\n'
            '    "    if gate_tol is None and fracture.get(\\"kind\\") == \\"mrst\\" and exchange.get(\\"kind\\") == \\"mrst_nnc\\":\\n",\n'
            '    "        gate_tol = 1.0e-10\\n",\n'
            '    "    gate = \\"\\"\\n",\n'
            '    "    if gate_tol is not None:\\n",\n'
            '    "        ok = bool(residual.size == 0 or np.nanmax(np.abs(residual)) <= gate_tol)\\n",\n'
            '    "        gate = f\\" gate={\'OK\' if ok else \'CHECK\'}(tol={gate_tol:.0e})\\"\\n",\n'
            '    "    display = fracture_balance_display_label(label, fracture)\\n",\n'
            '    "    print(f\\"  {display:<20} sum(Q_out incl. tips)+E_matrix: {exchange_residual_stats(residual)}{gate}\\")\\n",\n'
            '    "    return residual\\n",\n',
            '    "def print_fracture_single_phase_balance(label, fracture, exchange):\\n",\n'
            '    "    residual = fracture_single_phase_residual(fracture, exchange)\\n",\n'
            '    "    print(f\\"  {label:<10} fracture cells sum(Q_out)+E_matrix: {exchange_residual_stats(residual)}\\")\\n",\n'
            '    "    return residual\\n",\n',
            "fracture balance diagnostic",
            1,
        ),
        (
            '    "    required = [\\"nnc_mat_cell\\", \\"nnc_flux_m2f\\", \\"nnc_s\\"]\\n",\n',
            '    "    required = [\\"nnc_mat_cell\\", \\"nnc_flux_m2f\\"]\\n",\n',
            "MRST NNC required fields",
            1,
        ),
        (
            '    "    nnc_s = np.asarray(mrst[\\"nnc_s\\"], dtype=float).reshape(-1)\\n",\n'
            '    "    nnc_frac_cell = np.asarray(mrst.get(\\"nnc_frac_cell\\", np.full_like(mat_cell, -1)), dtype=np.int64).reshape(-1)\\n",\n',
            '    "    nnc_s = np.asarray(mrst.get(\\"nnc_s\\", np.full_like(flux_m2f, np.nan)), dtype=float).reshape(-1)\\n",\n',
            "MRST NNC coordinate fallback",
            1,
        ),
        (
            '    "    conn_cv_ids, conn_E, conn_s, conn_mrst_frac_cell = [], [], [], []\\n",\n'
            '    "    for midx, q_m2f, sval, fidx in zip(mat_cell, flux_m2f, nnc_s, nnc_frac_cell):\\n",\n',
            '    "    conn_cv_ids, conn_E, conn_s = [], [], []\\n",\n'
            '    "    for midx, q_m2f, sval in zip(mat_cell, flux_m2f, nnc_s):\\n",\n',
            "MRST NNC loop signature",
            1,
        ),
        (
            '    "        if not np.isfinite(sval):\\n",\n'
            '    "            raise RuntimeError(\\"MRST NNC exchange contains a non-finite nnc_s value.\\")\\n",\n'
            '    "        conn_s.append(float(sval))\\n",\n'
            '    "        conn_mrst_frac_cell.append(int(fidx))\\n",\n',
            '    "        conn_s.append(float(sval) if np.isfinite(sval) else np.nan)\\n",\n',
            "MRST NNC coordinate append",
            1,
        ),
        (
            '    "        \\"conn_s\\": np.asarray(conn_s, dtype=float),\\n",\n'
            '    "        \\"conn_mrst_frac_cell\\": np.asarray(conn_mrst_frac_cell, dtype=np.int64),\\n",\n',
            '    "        \\"conn_s\\": np.asarray(conn_s, dtype=float),\\n",\n',
            "MRST native fracture-cell debug field",
            1,
        ),
        (
            '    "_mrst_conn_s = np.asarray(transport_rect_exchanges[\\"MRST\\"].get(\\"conn_s\\", []), dtype=float)\\n",\n'
            '    "_mrst_conn_frac = np.asarray(transport_rect_exchanges[\\"MRST\\"].get(\\"conn_frac_ids\\", []), dtype=np.int32)\\n",\n'
            '    "_mrst_native_frac = np.asarray(transport_rect_exchanges[\\"MRST\\"].get(\\"conn_mrst_frac_cell\\", []), dtype=np.int64)\\n",\n'
            '    "_mrst_valid_conn_s = np.isfinite(_mrst_conn_s)\\n",\n'
            '    "_mrst_n_mapped = int(np.count_nonzero(_mrst_valid_conn_s & (_mrst_conn_frac >= 0)))\\n",\n'
            '    "_mrst_mapping_check = \\"\\"\\n",\n'
            '    "if len(_mrst_native_frac) == len(_mrst_conn_frac) and len(_mrst_native_frac) and np.all(_mrst_native_frac > 0):\\n",\n'
            '    "    _mrst_native_local = _mrst_native_frac - int(np.min(_mrst_native_frac))\\n",\n'
            '    "    _mrst_n_match = int(np.count_nonzero(_mrst_native_local == _mrst_conn_frac))\\n",\n'
            '    "    _mrst_mapping_check = f\\"; {_mrst_n_match}/{len(_mrst_conn_frac)} agree with native nnc_frac_cell order\\"\\n",\n'
            '    "print(\\n",\n'
            '    "    f\\"MRST NNC exchange mapping: {_mrst_n_mapped}/{len(_mrst_conn_s)} native connections \\"\\n",\n'
            '    "    f\\"mapped to fracture cells by per-connection nnc_s{_mrst_mapping_check}\\"\\n",\n'
            '    ")\\n",\n'
            '    "\\n",\n',
            "",
            "MRST NNC mapping diagnostic print",
            1,
        ),
    ]

    for old, new, label, expected in replacements:
        text = replace_checked(text, old, new, label, expected)

    json.loads(text)
    notebook.write_text(text)
    print(f"Restored aperture-scaled checkpoint in {notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
