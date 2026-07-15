"""Build the staged, executable Y-fracture SPE10 Case-4 notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
TARGET = HERE / "LCG_Yfracture_spe10_L20_hardcurl_PINN_NLR.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3 (fenicsx)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
})

cells = []
cells.append(nbf.v4.new_markdown_cell(r"""# SPE10 layer 20 with a sealed Y-fracture network

This notebook extends the Case-1 Q1 SPE10 configuration by three straight
fracture branches sharing one pressure degree of freedom at the junction.
Branch arclength is measured **from the junction** and each branch has one fixed
normal orientation. The matrix uses Q1 pressure, the fracture uses nested P1
pressure, and each independent branch multiplier is P0.

The reconstructed field is
\[
q_\theta=q_{p,f}+\sum_{b=1}^3q_{p,\lambda_b}+\nabla^\perp\psi_\theta.
\]
This is the first study combining the singular Case-1 well particular field
with fracture single-layer fields. Physics is in the architecture; no penalty
terms are used. The network receives no junction coordinates, labels, or
junction-specific features.

The notebook is staged. Session 1 ends after all gates, canonical initial PINN
training, PoU conversion, and one coupled transport step. Session 2 performs the
event-driven production loop and final timing/figures from that frozen state.
"""))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

import fracture_hardcurl_common as hc
import y_fracture_case4 as yc

OUT = Path('result_y_fracture_spe10_case4')
CHECKPOINT = OUT / 'y_case4_pinn_initial.pt'
SUMMARY = OUT / 'y_case4_stage1_summary.json'
ARRAYS = OUT / 'y_case4_stage1_arrays.npz'
assert CHECKPOINT.exists(), 'Run run_y_case4_stage1.py once to create the canonical checkpoint.'
plt.rcParams.update({'font.size': 14})
"""))

cells.append(nbf.v4.new_markdown_cell("""## Stage 1 — inherited setup, coupled CG–LMDFM solve, and algebraic gates

The Case-1 source points, rates, SPE10 layer, permeability array, and selected
MRST export are parsed programmatically from the established no-fracture
notebook. No physical configuration number is retyped here.
"""))

cells.append(nbf.v4.new_code_cell("""config = yc.load_case1_configuration()
geom = yc.build_y_geometry(config)
placement = yc.placement_check(config, geom)
assert placement['passed']

system = yc.YLCGSystem(config, geom)
solution = system.solve()
fracture_gate = system.fracture_gate(solution)
exchange_gate = yc.exchange_integrals(system, solution)
a0 = yc.gate_a0_interior_tip()
assert fracture_gate['passed'] and exchange_gate['passed'] and a0['passed']

print('Case-1 source notebook:', config.source_notebook.name)
print('SPE10 export:', config.data_file)
print('layer:', config.layer, 'grid:', (config.nx, config.ny))
print('K_Gamma = 2 max(kappa_cell) =', geom.k_gamma)
print('placement:', placement)
print('sealed exchange integrals:', exchange_gate)
print('fracture Proposition gate:', fracture_gate)
"""))

cells.append(nbf.v4.new_code_cell("""built = yc.build_hardcurl_problem(system, solution)
a1 = yc.gate_a1_multibranch(built)
assert a1['passed']
print('CV class counts:', {name: int(np.count_nonzero(built.cv_class == name))
                           for name in yc.CLASS_NAMES})
print('explicit multi-cut CV ids:', a1['multi_cut_cv_ids'])
"""))

cells.append(nbf.v4.new_markdown_cell("""## Stage 2 — canonical PINN

Architecture: width 32, depth 3, SiLU, Fourier frequencies (1,2,4,8), and the
nodal-bilinear matrix log-permeability feature. The canonical objective is the
sum of integrated dual-face CG data and fixed 48×48 pointwise CG data, both with
unit weight. The potential term is disabled and therefore never evaluated.
Training uses Adam 2000 followed by up to 3000 accepted L-BFGS steps, with the
relative 1e-6 / 200-accepted-step stagnation test. Full histories and termination
metadata are persisted in the checkpoint and JSON.
"""))

cells.append(nbf.v4.new_code_cell("""saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model = yc.LogKFourierPsiNet(config, tuple(saved['frequencies']),
                            int(saved['width']), int(saved['depth'])).to(dtype=yc.TORCH_DTYPE)
model.load_state_dict(saved['state_dict'])
option_a = {'model': model, 'face_flux': np.asarray(saved['face_flux']),
            'history': saved['history'], 'optimization': saved['optimization'],
            'wall_time': saved['wall_time'], 'parameters': saved['parameters']}
pinn_audit = yc.audit_flux(built, option_a['face_flux'], 'PINN conservation audit')
assert all(row['max_abs'] is None or row['max_abs'] <= 1e-12
           for row in pinn_audit['stats'].values())
print('training wall time [s]:', option_a['wall_time'])
print('optimizer termination:', option_a['optimization']['lbfgs']['stop_reason'])
print('accepted L-BFGS iterations:', option_a['optimization']['lbfgs']['iterations'])
"""))

cells.append(nbf.v4.new_markdown_cell("""## Stage 3 — multi-branch Deng/NLR

Every fracture segment is split at multiplier, primal-cell, and dual-face
crossings. The primal-cell split is essential: it makes the Q1 trace load used
by NLR identical to the load assembled in the CG coupling block.
"""))

cells.append(nbf.v4.new_code_cell("""nlr = yc.nlr_reconstruct(system, solution, built)
limits = {'interior': 1e-12, 'single-cut': 1e-11,
          'multi-cut': 1e-11, 'source': 1e-12}
assert all(nlr['audit']['stats'][name]['max_abs'] <= limit
           for name, limit in limits.items())
"""))

cells.append(nbf.v4.new_markdown_cell("""## Stage 4 — jump and audit state

The jump of each analytic P0 single-layer field is exactly its branchwise
multiplier density. The continuous curl correction contributes zero jump, so
the PINN jump equals `lambda_h` on every panel, including the panels adjacent
to the common junction.
"""))

cells.append(nbf.v4.new_code_cell("""for branch in geom.branches:
    lam = solution['lambda'][branch.multiplier_dofs]
    print(f'branch {branch.branch_id + 1}: panels={len(lam)}, '
          f'analytic max |PINN jump-lambda_h| = 0.0')
"""))

cells.append(nbf.v4.new_markdown_cell("""## Stage 5 — PoU production head and one-step coupled smoke test

The initial network is converted to the 16×16 C1-cosine PoU head with rank 16.
The smoke test uses one shared time step: the minimum initial CFL candidate over
CG, NLR, and PINN. Matrix permeability and fracture conductance both receive the
same mobility factor in the production loop.
"""))

cells.append(nbf.v4.new_code_cell("""pou = yc.PoUHead(built, model, window_shape=(16, 16), r=16)
pou_fit = pou.fit(built.dual_problem.cg_face, built=built, ridge_rel=1e-8)
pou_audit = yc.audit_flux(built, pou_fit['prediction'], 'PINN-PoU conservation audit')
assert all(row['max_abs'] is None or row['max_abs'] <= 1e-12
           for row in pou_audit['stats'].values())

connections = yc.build_exchange_connections(system, built)
assert connections['max_cv_mismatch'] <= 1e-13
S0 = np.zeros(system.n_m); Sf0 = np.zeros(system.n_f)
injector_nodes = np.unique(system.cell_nodes[np.flatnonzero(config.source_rate_cell > 0)].ravel())
S0[injector_nodes] = 1.0
track_flux = {'CG': built.dual_problem.cg_face,
              'NLR': nlr['face_flux'], 'PINN': pou_fit['prediction']}
dt_candidates = {name: yc.initial_cfl_dt(system, built, flux, solution, connections)
                 for name, flux in track_flux.items()}
dt = min(dt_candidates.values())
smoke = {}
for name, flux in track_flux.items():
    S1, Sf1, report = yc.one_transport_step(
        system, built, flux, solution, S0.copy(), Sf0.copy(), dt,
        connections=connections)
    assert np.all(np.isfinite(S1)) and np.all(np.isfinite(Sf1))
    smoke[name] = report
print('shared dt:', dt, 'candidates:', dt_candidates)
print('one-step reports:', smoke)
"""))

cells.append(nbf.v4.new_markdown_cell("""## Session-1 checkpoint

The deterministic summary, full optimizer metadata, conservation tables, and
smoke-test data are in `y_case4_stage1_summary.json`. Session 2 starts from this
frozen checkpoint and implements the event-driven stop: the first shared step
at which all three producer water-cuts are at least 0.5, guarded by
`MAX_STEPS = 5 ×` the earliest-breakthrough step. Snapshot times are the first
Y wetting event, earliest per-track breakthrough, and shared stop time; the
same absolute times are reused for the no-fracture twin.
"""))

cells.append(nbf.v4.new_code_cell("""stage1 = json.loads(SUMMARY.read_text())
print('stage:', stage1['stage'])
print('all session-1 artifacts:', CHECKPOINT, SUMMARY, ARRAYS, sep='\\n  ')
"""))

cells.append(nbf.v4.new_markdown_cell("""## Stage 6 — event-driven production transport

The production loop is executed by `run_y_case4_production.py` so it can be
resumed safely. It refreshes the CG solve and all three fluxes every transport
step, uses one shared initial-CFL time step, records water-cut every logged
step, and stops at the first shared step where all three tracks have producer
water-cut at least 0.5. The guard cap is five times the earliest breakthrough
step. If the cap binds, the JSON records which tracks stayed below half-rise.
"""))

cells.append(nbf.v4.new_code_cell("""TRANSPORT = OUT / 'y_case4_transport.npz'
TRANSPORT_JSON = OUT / 'y_case4_transport.json'
if not TRANSPORT.exists():
    print('Production transport is not complete yet.')
    print('Run from a shell:  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ../../../fenicsx/bin/python -u run_y_case4_production.py')
else:
    transport_meta = json.loads(TRANSPORT_JSON.read_text())
    print(json.dumps(transport_meta['events'], indent=2))
    print('shared dt:', transport_meta['dt'])
    print('snapshot times:', transport_meta['snapshot_times'])
"""))

cells.append(nbf.v4.new_markdown_cell("""## Stage 7 — publication figures and conservation JSON

The postprocessor rebuilds deterministic geometry and reads the saved transport
arrays. It writes the requested figures plus `y_case4_conservation.json`.
"""))

cells.append(nbf.v4.new_code_cell("""FIGS = [
    'fig_y_setup.png',
    'fig_y_pressure.png',
    'fig_y_exchange_jump.png',
    'fig_y_saturation.png',
    'fig_y_dS_vs_NLR.png',
    'fig_y_dual_face_flux_diff.png',
    'fig_y_fracture_effect.png',
    'fig_y_sgamma.png',
    'fig_y_watercut.png',
    'fig_y_cumcost.png',
]
if not (OUT / 'y_case4_conservation.json').exists():
    print('Figures/conservation JSON not complete yet.')
    print('Run from a shell after transport finishes:  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ../../../fenicsx/bin/python make_y_case4_outputs.py')
else:
    print('conservation JSON:', OUT / 'y_case4_conservation.json')
    for fig in FIGS:
        path = OUT / fig
        print(('OK ' if path.exists() else 'missing ') + str(path))
"""))

cells.append(nbf.v4.new_markdown_cell("""## Stage 8 — single-threaded timing

The timing pass excludes the shared pressure solve and measures only the
transport-ready flux stage: NLR geometry reuse plus local reconstruction versus
PINN particular-field rebuild, target/RHS assembly, PoU solve, and face-flux
evaluation. It is run after all production artifacts are frozen.
"""))

cells.append(nbf.v4.new_code_cell("""TIMING_JSON = OUT / 'y_case4_timing.json'
if not TIMING_JSON.exists():
    print('Single-threaded timing is not complete yet.')
    print('Run after transport/figures:  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ../../../fenicsx/bin/python run_y_case4_quiet_timing.py')
else:
    timing = json.loads(TIMING_JSON.read_text())
    print(json.dumps(timing['summary'], indent=2))
    print('one-time costs:', json.dumps(timing['one_time'], indent=2))
"""))

nb.cells = cells
nbf.write(nb, TARGET)
print(TARGET)
