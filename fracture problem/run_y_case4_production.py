"""Event-driven coupled production loop for SPE10 Y-fracture Case 4."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import y_fracture_case4 as yc


HERE=Path(__file__).resolve().parent
OUT=HERE/"result_y_fracture_spe10_case4";OUT.mkdir(exist_ok=True)
PINN_CKPT=OUT/"y_case4_pinn_initial.pt"
RUN_CKPT=OUT/"y_case4_production_resume.pt"
TRANSPORT_NPZ=OUT/"y_case4_transport.npz"
TRANSPORT_JSON=OUT/"y_case4_transport.json"
LOG_EVERY=10;CHECKPOINT_EVERY=100
BT_LEVEL=1.0e-3;HALF_LEVEL=0.5;CFL_SAFETY=0.45
ABSOLUTE_MAX_STEPS=200000


def jsonable(v):
    if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [jsonable(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,(np.floating,np.integer,np.bool_)):return v.item()
    return v


def nofracture_view(template:yc.BuiltProblem,solution:dict,cg_face:np.ndarray)->yc.BuiltProblem:
    p0=template.dual_problem;problem=SimpleNamespace(**p0.__dict__)
    zcv=np.zeros_like(p0.cv_lambda_h);zface=np.zeros_like(p0.qpl_h_face)
    problem.ctx=solution;problem.cv_lambda_h=zcv;problem.cv_lambda_exact=zcv.copy()
    problem.qpl_h_face=zface;problem.qpl_exact_face=zface.copy();problem.cg_face=cg_face
    problem.lambda_h_density=np.zeros_like(p0.lambda_h_density)
    problem.lambda_h_nodal_density=np.zeros_like(p0.lambda_h_nodal_density)
    return yc.BuiltProblem(problem,template.dual,template.cv_class,
                           template.cv_cut_count,template.source_mask,
                           template.boundary_mask,{},template.qpf)


def residual_max(built:yc.BuiltProblem,flux:np.ndarray,cv_lambda:np.ndarray)->float:
    r=yc.scatter_flux(built.dual,np.asarray(flux))-built.dual_problem.cv_source-cv_lambda
    return float(np.max(np.abs(r)))


def save_resume(payload):
    tmp=RUN_CKPT.with_suffix(".tmp")
    torch.save(payload,tmp);tmp.replace(RUN_CKPT)


parser=argparse.ArgumentParser()
parser.add_argument("--pilot-steps",type=int,default=0)
parser.add_argument("--fresh",action="store_true")
args=parser.parse_args()

config=yc.load_case1_configuration();geom=yc.build_y_geometry(config)
system=yc.YLCGSystem(config,geom);solution0=system.solve()
built0=yc.build_hardcurl_problem(system,solution0)
dynamic=yc.YDynamicGeometry(system,built0);fast_nlr=yc.FastNLR(system,dynamic)

saved=torch.load(PINN_CKPT,map_location="cpu",weights_only=False)
model=yc.LogKFourierPsiNet(config,tuple(saved["frequencies"]),int(saved["width"]),
                           int(saved["depth"])).to(dtype=yc.TORCH_DTYPE)
model.load_state_dict(saved["state_dict"]);model.eval()
pou=yc.PoUHead(built0,model,window_shape=(16,16),r=16)
pou.factorize(1.0e-8)

inj=np.unique(system.cell_nodes[np.flatnonzero(config.source_rate_cell>0)].ravel())
cut_mask=built0.cv_cut_count>0
track_names=("CG","NLR","PINN")

if RUN_CKPT.exists() and not args.fresh and not args.pilot_steps:
    state=torch.load(RUN_CKPT,map_location="cpu",weights_only=False)
    step=int(state["step"]);tracks=state["tracks"];curves=state["curves"]
    events=state["events"];snapshots=state["snapshots"];timing=state["timing"]
    pou.theta=np.asarray(state["pou_theta"])
    print("resumed production at step",step)
else:
    step=0
    tracks={name:{"S":np.zeros(system.n_m),"Sf":np.zeros(system.n_f)} for name in track_names}
    for tr in tracks.values():tr["S"][inj]=1.0
    curves={"step":[],"time":[],**{f"wc_{n}":[] for n in track_names},
            **{f"rmse_{n}":[] for n in track_names},
            **{f"cfl_{n}":[] for n in track_names},
            **{f"Rmax_{n}":[] for n in track_names}}
    events={"first_y_step":None,"earliest_breakthrough_step":None,
            "breakthrough_step":{n:None for n in track_names},
            "half_rise_step":{n:None for n in track_names},
            "dynamic_max_steps":None,"stop_reason":None}
    snapshots={};timing={n:{"solve":[],"flux":[],"transport":[]} for n in track_names}

# One common initial dt from all three initial flux fields.
initial_flux={}
initial_conn=dynamic.connections(solution0)
initial_flux["CG"]=dynamic.cg_face(solution0)
initial_flux["NLR"]=fast_nlr.reconstruct(solution0)["face_flux"]
view0=dynamic.step_view(solution0,initial_flux["CG"])
theta_before_dt=pou.theta.copy()
initial_flux["PINN"]=pou.fit(initial_flux["CG"],built=view0)["prediction"]
pou.theta=theta_before_dt
dt_candidates={n:yc.initial_cfl_dt(system,built0,initial_flux[n],solution0,initial_conn,
                                    cfl=CFL_SAFETY) for n in track_names}
dt=min(dt_candidates.values())
print("dynamic build [s]",dynamic.build_s,"FastNLR build [s]",fast_nlr.build_s)
print("shared dt",dt,"candidates",dt_candidates)

pilot_limit=step+args.pilot_steps if args.pilot_steps else None
production_t0=time.perf_counter()
while True:
    step+=1;step_rows={};post={}
    for name in track_names:
        S=tracks[name]["S"];Sf=tracks[name]["Sf"]
        mob=yc.mobility_factor(yc.project_dual_to_cells(system,S))
        mobf=yc.mobility_factor(Sf)
        ts=time.perf_counter();sol=system.solve(mob,mobf);solve_s=time.perf_counter()-ts
        conn=dynamic.connections(sol)
        if conn["max_cv_mismatch"]>1.0e-12:
            raise RuntimeError(f"exchange mapping failed at step {step}: {conn['max_cv_mismatch']}")

        tf=time.perf_counter()
        if name=="CG":
            cg=dynamic.cg_face(sol);flux=cg
        elif name=="NLR":
            out=fast_nlr.reconstruct(sol);flux=out["face_flux"]
            flux_s=time.perf_counter()-tf
            cg=dynamic.cg_face(sol)
        else:
            cg=dynamic.cg_face(sol);qpl=dynamic.qpl_face(sol)
            view=dynamic.step_view(sol,cg,qpl)
            fit=pou.fit(cg,built=view,anchor=pou.theta,ridge_rel=1.0e-8)
            flux=fit["prediction"]
        if name!="NLR":flux_s=time.perf_counter()-tf

        ttr=time.perf_counter()
        S1,Sf1,report=yc.one_transport_step(system,built0,flux,sol,S,Sf,dt,
                                             connections=conn)
        transport_s=time.perf_counter()-ttr
        cvlam=dynamic.cv_lambda(sol)
        rmax=residual_max(built0,flux,cvlam)
        limit=yc.initial_cfl_dt(system,built0,flux,sol,conn,cfl=CFL_SAFETY)
        cfl_value=CFL_SAFETY*dt/max(limit,1.0e-300)
        rmse=float(np.sqrt(np.mean((flux-cg)**2)))
        post[name]={"S":S1,"Sf":Sf1,"wc":float(report["producer_watercut"]),
                    "rmse":rmse,"cfl":cfl_value,"Rmax":rmax}
        timing[name]["solve"].append(solve_s);timing[name]["flux"].append(flux_s)
        timing[name]["transport"].append(transport_s)
        step_rows[name]=(solve_s,flux_s,transport_s)
    tracks={n:{"S":post[n]["S"],"Sf":post[n]["Sf"]} for n in track_names}

    for name in track_names:
        wc=post[name]["wc"]
        if events["breakthrough_step"][name] is None and wc>BT_LEVEL:
            events["breakthrough_step"][name]=step
        if events["half_rise_step"][name] is None and wc>=HALF_LEVEL:
            events["half_rise_step"][name]=step
    if events["earliest_breakthrough_step"] is None:
        vals=[v for v in events["breakthrough_step"].values() if v is not None]
        if vals:
            events["earliest_breakthrough_step"]=min(vals)
            events["dynamic_max_steps"]=min(ABSOLUTE_MAX_STEPS,5*min(vals))
            snapshots["earliest_breakthrough"]={
                "step":step,"time":step*dt,
                **{f"S_{n}":tracks[n]["S"].copy() for n in track_names},
                **{f"Sf_{n}":tracks[n]["Sf"].copy() for n in track_names},
            }
    if events["first_y_step"] is None and any(np.max(tracks[n]["S"][cut_mask])>BT_LEVEL
                                               for n in track_names):
        events["first_y_step"]=step
        snapshots["first_y"]={
            "step":step,"time":step*dt,
            **{f"S_{n}":tracks[n]["S"].copy() for n in track_names},
            **{f"Sf_{n}":tracks[n]["Sf"].copy() for n in track_names},
        }

    if step%LOG_EVERY==0 or step==1:
        curves["step"].append(step);curves["time"].append(step*dt)
        for name in track_names:
            for key in ("wc","rmse","cfl","Rmax"):
                curves[f"{key}_{name}"].append(post[name][key])
    if step==1 or step%100==0:
        msg=" ".join(f"{n}:wc={post[n]['wc']:.3e},cfl={post[n]['cfl']:.3f},"
                     f"wall={sum(step_rows[n]):.3f}s" for n in track_names)
        print(f"step={step} T={step*dt:.6g} {msg}",flush=True)

    all_half=all(v is not None for v in events["half_rise_step"].values())
    cap=events["dynamic_max_steps"]
    if all_half:
        events["stop_reason"]="all_tracks_watercut_at_least_0.5";break
    if cap is not None and step>=cap:
        events["stop_reason"]="dynamic_max_steps_bound";break
    if step>=ABSOLUTE_MAX_STEPS:
        events["stop_reason"]="absolute_max_steps_bound";break
    if pilot_limit is not None and step>=pilot_limit:
        events["stop_reason"]="pilot_limit";break
    if not args.pilot_steps and step%CHECKPOINT_EVERY==0:
        save_resume({"step":step,"tracks":tracks,"curves":curves,"events":events,
                     "snapshots":snapshots,"timing":timing,"pou_theta":pou.theta})

snapshots["stop"]={"step":step,"time":step*dt,
                    **{f"S_{n}":tracks[n]["S"].copy() for n in track_names},
                    **{f"Sf_{n}":tracks[n]["Sf"].copy() for n in track_names}}
print("fractured loop stop",events,"wall",time.perf_counter()-production_t0)

if args.pilot_steps:
    print("PILOT COMPLETE; production files were not replaced")
    raise SystemExit(0)

# No-fracture PINN twin, using the same dt and the three exact event steps.
nf=yc.NoFractureQ1System(system);Snf=np.zeros(system.n_m);Snf[inj]=1.0
nf0=nf.solve(yc.mobility_factor(yc.project_dual_to_cells(system,Snf)))
cg_nf0=dynamic.cg_face(nf0);view_nf0=nofracture_view(built0,nf0,cg_nf0)
pou_nf=yc.PoUHead(view_nf0,model,window_shape=(16,16),r=16);pou_nf.factorize(1.0e-8)
event_steps={}
for label,row in snapshots.items():
    event_steps.setdefault(int(row["step"]),[]).append(label)
nf_snapshots={};nf_curve_step=[];nf_curve_time=[];nf_curve_wc=[]
for k in range(1,step+1):
    mob=yc.mobility_factor(yc.project_dual_to_cells(system,Snf));sol=nf.solve(mob)
    cg=dynamic.cg_face(sol);view=nofracture_view(built0,sol,cg)
    flux=pou_nf.fit(cg,built=view,anchor=pou_nf.theta,ridge_rel=1.0e-8)["prediction"]
    Snf,rep=yc.one_matrix_transport_step(system,built0,flux,Snf,dt)
    if k%LOG_EVERY==0 or k==1:
        nf_curve_step.append(k);nf_curve_time.append(k*dt);nf_curve_wc.append(rep["producer_watercut"])
    if k in event_steps:
        for label in event_steps[k]:
            nf_snapshots[label]=Snf.copy()
    if k==1 or k%500==0:print("twin step",k,"of",step,flush=True)

events["final_step"]=step;events["final_time"]=step*dt
events["breakthrough_time"]={n:(None if v is None else v*dt)
                              for n,v in events["breakthrough_step"].items()}
events["half_rise_time"]={n:(None if v is None else v*dt)
                          for n,v in events["half_rise_step"].items()}
events["tracks_below_half_at_stop"]=[n for n,v in events["half_rise_step"].items() if v is None]

arrays={"dt":np.array(dt),"log_step":np.asarray(curves["step"]),
        "log_time":np.asarray(curves["time"]),
        "twin_log_step":np.asarray(nf_curve_step),"twin_log_time":np.asarray(nf_curve_time),
        "twin_watercut":np.asarray(nf_curve_wc)}
for key,val in curves.items():
    if key not in ("step","time"):arrays[key]=np.asarray(val)
for event,row in snapshots.items():
    arrays[f"snapshot_{event}_step"]=np.array(row["step"])
    arrays[f"snapshot_{event}_time"]=np.array(row["time"])
    for name in track_names:
        arrays[f"snapshot_{event}_S_{name}"]=row[f"S_{name}"]
        arrays[f"snapshot_{event}_Sf_{name}"]=row[f"Sf_{name}"]
    if event in nf_snapshots:arrays[f"snapshot_{event}_S_twin"]=nf_snapshots[event]
np.savez_compressed(TRANSPORT_NPZ,**arrays)

def stats(values):
    a=np.asarray(values);return {"mean":float(np.mean(a)),"median":float(np.median(a)),
                                 "p95":float(np.quantile(a,0.95)),"n":int(len(a))}
meta={"events":events,"dt":dt,"dt_candidates":dt_candidates,"CFL_safety":CFL_SAFETY,
      "BT_level":BT_LEVEL,"half_rise_level":HALF_LEVEL,
      "snapshot_times":{k:v["time"] for k,v in snapshots.items()},
      "timing_development_run":{n:{k:stats(v) for k,v in timing[n].items()} for n in track_names},
      "one_time":{"PINN_training_s":float(saved["wall_time"]),
                  "PoU_build_s":pou.build_s,"PoU_factor_s":pou._factor["factor_s"],
                  "NLR_geometry_s":fast_nlr.build_s,"dynamic_geometry_s":dynamic.build_s},
      "machine":{"platform":platform.platform(),"processor":platform.processor(),
                 "single_threaded_timing_pending":True},
      "outputs":{"npz":str(TRANSPORT_NPZ)}}
TRANSPORT_JSON.write_text(json.dumps(jsonable(meta),indent=2)+"\n")
if RUN_CKPT.exists():RUN_CKPT.unlink()
print("saved",TRANSPORT_NPZ);print("saved",TRANSPORT_JSON)
