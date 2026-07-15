"""Tip-B diagnostics for the KM case-1 fracture transport notebook.

This file is executed from the notebook after Stage 7.  It deliberately uses
the live notebook objects so that the printed arrays are exactly those used by
the transport run rather than independently reconstructed approximations.
"""


def run_tip_transport_diagnostics(ns):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    g = ns
    fracture = g["fracture_lcg"]
    exchange = g["exchange_lcg"]
    q_elem = np.asarray(fracture["Q_element"], dtype=float)
    s_nodes = np.asarray(fracture["s_centers"], dtype=float)
    h_bulk = float(g["h"])
    real_points = g["real_points_np"]

    # --------------------------------------------------------------
    # D1: last fracture-element fluxes and their bulk-grid crossings.
    # --------------------------------------------------------------
    n_tail = min(20, len(q_elem))
    tail_rows = []
    for elem in range(len(q_elem) - n_tail, len(q_elem)):
        s0, s1 = float(s_nodes[elem]), float(s_nodes[elem + 1])
        sm = 0.5 * (s0 + s1)
        xy = real_points([sm])[0]
        p0, p1 = real_points([s0, s1])
        ix = min(int(np.floor(xy[0] / h_bulk)), int(round(1.0 / h_bulk)) - 1)
        iy = min(int(np.floor(xy[1] / h_bulk)), int(round(1.0 / h_bulk)) - 1)
        cross_x = int(np.floor((p1[0] + 1.0e-13) / h_bulk)) != int(
            np.floor((p0[0] - 1.0e-13) / h_bulk)
        )
        cross_y = int(np.floor((p1[1] + 1.0e-13) / h_bulk)) != int(
            np.floor((p0[1] - 1.0e-13) / h_bulk)
        )
        tail_rows.append({
            "element": int(elem), "s_mid": sm, "x_mid": float(xy[0]),
            "y_mid": float(xy[1]), "bulk_ix": ix, "bulk_iy": iy,
            "crosses_x_gridline": bool(cross_x),
            "crosses_y_gridline": bool(cross_y),
            "q_elem": float(q_elem[elem]),
        })

    q_tail = q_elem[-n_tail:]
    q_smooth = np.convolve(np.pad(q_tail, 1, mode="edge"),
                           np.ones(3) / 3.0, mode="valid")
    q_osc = q_tail - q_smooth
    lag1 = float(np.corrcoef(q_osc[:-1], q_osc[1:])[0, 1])
    lag2 = float(np.corrcoef(q_osc[:-2], q_osc[2:])[0, 1])
    crossing_mask = np.asarray([
        row["crosses_x_gridline"] or row["crosses_y_gridline"]
        for row in tail_rows
    ], dtype=bool)
    crossing_abs_osc = float(np.mean(np.abs(q_osc[crossing_mask]))) \
        if np.any(crossing_mask) else float("nan")
    noncrossing_abs_osc = float(np.mean(np.abs(q_osc[~crossing_mask]))) \
        if np.any(~crossing_mask) else float("nan")

    print("\nTIP DIAGNOSTIC D1: q_elem near tip B")
    print(" elem      s_mid       bulk(i,j)  cross-x cross-y          q_elem")
    for row in tail_rows:
        print(
            f" {row['element']:4d}  {row['s_mid']:.9f}  "
            f"({row['bulk_ix']:2d},{row['bulk_iy']:2d})      "
            f"{int(row['crosses_x_gridline'])}       "
            f"{int(row['crosses_y_gridline'])}    {row['q_elem']:+.10e}"
        )
    print(
        f" detrended q correlations: lag-1={lag1:+.4f}, lag-2={lag2:+.4f}; "
        f"mean |osc| crossing={crossing_abs_osc:.3e}, "
        f"noncrossing={noncrossing_abs_osc:.3e}"
    )

    # --------------------------------------------------------------
    # D2: actual exchange deposited into the fine fracture CVs.
    # --------------------------------------------------------------
    e_by_frac = np.bincount(
        exchange["conn_frac_ids"], weights=exchange["conn_E"],
        minlength=len(fracture["length"]),
    )
    e_density = e_by_frac / np.asarray(fracture["length"], dtype=float)
    lam_edges = np.asarray(g["_LAM_S_NODES"], dtype=float)
    print("\nTIP DIAGNOSTIC D2: exchange actually used by the last 10 fracture CVs")
    print(" fracCV      s_center    lambda-panel      length             E_CV       E_CV/length")
    exchange_tail_rows = []
    for fi in range(max(0, len(e_by_frac) - 10), len(e_by_frac)):
        panel = int(np.clip(
            np.searchsorted(lam_edges, s_nodes[fi], side="right") - 1,
            0, len(lam_edges) - 2,
        ))
        row = {
            "fracture_cv": int(fi), "s_center": float(s_nodes[fi]),
            "lambda_panel": panel, "length": float(fracture["length"][fi]),
            "exchange": float(e_by_frac[fi]),
            "exchange_density": float(e_density[fi]),
        }
        exchange_tail_rows.append(row)
        print(
            f" {fi:6d}  {s_nodes[fi]:.9f}      {panel:4d}      "
            f"{fracture['length'][fi]:.9e}  {e_by_frac[fi]:+.10e}  "
            f"{e_density[fi]:+.10e}"
        )

    # --------------------------------------------------------------
    # D3: multiplier-panel-scale fracture transport experiment.
    # Matrix CVs and matrix face fluxes are unchanged.  Lambda is integrated
    # exactly into its own P0 panels and the fracture face flux follows from
    # the conservative cumulative balance q_R-q_L+integral(lambda)=0.
    # --------------------------------------------------------------
    def build_multiplier_exchange():
        conn_cv, conn_frac, conn_e, conn_s = [], [], [], []
        for cid, subpolys in enumerate(g["problem"].cv_polys):
            for poly in subpolys:
                interval = g["hc"]._line_interval_in_polygon(
                    np.asarray(poly), g["FRAC_A"], g["tau_np"],
                    g["normal_np"], g["L_gamma"],
                )
                if interval is None:
                    continue
                for a, b in g["_split_interval_at_knots"](
                    interval[0], interval[1], lam_edges
                ):
                    mid = 0.5 * (a + b)
                    panel = int(np.clip(
                        np.searchsorted(lam_edges, mid, side="right") - 1,
                        0, len(lam_edges) - 2,
                    ))
                    half = 0.5 * (b - a)
                    sq = mid + half * g["_GS2"]
                    value = float(half * np.sum(g["lambda_h_on_s"](sq)))
                    if abs(value) > g["TRANSPORT_EXCHANGE_TOL"]:
                        conn_cv.append(int(cid)); conn_frac.append(panel)
                        conn_e.append(value); conn_s.append(mid)
        conn_cv = np.asarray(conn_cv, dtype=np.int32)
        conn_frac = np.asarray(conn_frac, dtype=np.int32)
        conn_e = np.asarray(conn_e, dtype=float)
        return {
            "kind": "lcg_lambda_h_multiplier_panels",
            "conn_cv_ids": conn_cv, "conn_frac_ids": conn_frac,
            "conn_E": conn_e, "conn_s": np.asarray(conn_s, dtype=float),
            "E_by_cell": np.bincount(
                conn_cv, weights=conn_e, minlength=len(g["problem"].coords)
            ),
        }

    coarse_exchange = build_multiplier_exchange()
    coarse_e_by_frac = np.bincount(
        coarse_exchange["conn_frac_ids"], weights=coarse_exchange["conn_E"],
        minlength=len(lam_edges) - 1,
    )
    coarse_q_face = np.r_[0.0, -np.cumsum(coarse_e_by_frac)]
    tip_closure_before = float(coarse_q_face[-1])
    if abs(tip_closure_before) <= 1.0e-10:
        coarse_q_face[-1] = 0.0
    coarse_fracture = {
        "kind": "lcg_multiplier_scale_diagnostic",
        "grid": "P0 multiplier panels",
        "s_faces": lam_edges.copy(),
        "s_centers": 0.5 * (lam_edges[:-1] + lam_edges[1:]),
        "length": np.diff(lam_edges),
        "pore_volume": g["TRANSPORT_FRAC_POROSITY"] * np.diff(lam_edges),
        "Q_face": coarse_q_face,
        "tip_A_flux": 0.0, "tip_B_flux": -coarse_q_face[-1],
        "direction": g["fracture_direction_label"]({"Q_face": coarse_q_face}),
    }

    coarse_states = {}
    for name in ("CG", "NLR", "PINN"):
        template = {
            "oriented": g["tracks"][name]["oriented"],
            "fracture": coarse_fracture,
            "exchange": coarse_exchange,
            "S": np.zeros(len(g["dual_areas"]), dtype=float),
            "S_frac": np.zeros(len(coarse_fracture["length"]), dtype=float),
        }
        coarse_states[name] = g["advance_track_to_times"](
            template, g["transport_snapshot_times"], g["shared_dt"]
        )

    def roughness(values):
        values = np.asarray(values, dtype=float)
        return {
            "total_variation": float(np.sum(np.abs(np.diff(values)))),
            "second_difference_rms": float(np.sqrt(np.mean(np.diff(values, 2) ** 2)))
            if len(values) > 2 else 0.0,
            "last_cell": float(values[-1]),
            "last_minus_penultimate": float(values[-1] - values[-2]),
        }

    coarse_rows = {}
    for row, (pvi, tabs) in enumerate(zip(
        g["transport_snapshot_pvi"], g["transport_snapshot_times"]
    )):
        coarse_rows[f"{float(pvi):.1f}"] = {}
        for name in ("CG", "NLR", "PINN"):
            fine = g["snapshots"][float(tabs)][name]["S_frac"]
            coarse = coarse_states[name][float(tabs)]["S_frac"]
            coarse_rows[f"{float(pvi):.1f}"][name] = {
                "fine": roughness(fine), "multiplier_scale": roughness(coarse),
            }

    print("\nTIP DIAGNOSTIC D3: multiplier-scale fracture transport")
    print(f" panels={len(coarse_fracture['length'])}; cumulative tip closure before forcing zero="
          f"{tip_closure_before:+.3e}")
    print(json.dumps(coarse_rows, indent=2))

    fig, axes = plt.subplots(
        1, len(g["transport_snapshot_times"]), figsize=(14.4, 4.2),
        constrained_layout=True, sharey=True,
    )
    for row, (ax, pvi, tabs) in enumerate(zip(
        axes, g["transport_snapshot_pvi"], g["transport_snapshot_times"]
    )):
        for name, color in zip(("CG", "NLR", "PINN"), ("C0", "C1", "C2")):
            ax.plot(
                s_nodes, g["snapshots"][float(tabs)][name]["S_frac"],
                color=color, lw=1.0, alpha=0.45,
                label=f"{name}, node scale" if row == 0 else None,
            )
            ax.plot(
                coarse_fracture["s_centers"],
                coarse_states[name][float(tabs)]["S_frac"],
                color=color, lw=2.0, ls="--",
                label=f"{name}, multiplier scale" if row == 0 else None,
            )
        ax.set_title(rf"$T={float(pvi):.1f}$ (PVI)")
        ax.set_xlabel(r"fracture arc length $s$")
        ax.set_ylim(0.0, 1.0); ax.grid(False)
    axes[0].set_ylabel(r"$S_{w,\Gamma}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    outfig = g["OUTDIR"] / "fig_km_tip_multiplier_scale_diagnostic.png"
    fig.savefig(outfig, dpi=300, bbox_inches="tight")
    plt.show()

    report = {
        "q_elem_tip_rows": tail_rows,
        "q_periodicity": {
            "lag1_detrended_correlation": lag1,
            "lag2_detrended_correlation": lag2,
            "mean_abs_osc_on_crossing_elements": crossing_abs_osc,
            "mean_abs_osc_on_noncrossing_elements": noncrossing_abs_osc,
        },
        "exchange_tip_rows": exchange_tail_rows,
        "multiplier_scale": {
            "panel_count": int(len(coarse_fracture["length"])),
            "tip_closure_before_roundoff_zero": tip_closure_before,
            "rows": coarse_rows,
        },
    }
    outfile = g["OUTDIR"] / "km_case1_tip_transport_diagnostic.json"
    outfile.write_text(json.dumps(report, indent=2) + "\n")
    print("saved:", outfig)
    print("saved:", outfile)
    return report

