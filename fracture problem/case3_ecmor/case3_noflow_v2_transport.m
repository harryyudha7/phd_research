%% Case-1 EDFM sealed-tip reference v2 -- STAGE 2: 10-snapshot transport + full export.
%% Loads the stage-1 checkpoint (solve already verified sealed). Same explicit upwind scheme as v1,
%% run on the MRST/EDFM grid with the frozen MRST fluxes (matrix faces + fracture faces + NNC m2f).
%% 10 evenly spaced snapshots to PVI=1.0; matrix + fracture saturations; times in PVI AND absolute.
ckpt='c:\Users\muchamad\mrst-project\case3_noflow_v2_checkpoint.mat';
outfile='c:\Users\muchamad\mrst-project\case3_mrst_export_noflow_v2.mat';
load(ckpt);   % brings G,state,pv,nc,nfrac,ncell,T_final,Qright,PV_matrix,RIGHT_BC_S,CFL,FPRIME_MAX, all static export fields, meta_*, conventions
N=G.faces.neighbors;

%% unified connection list (matrix faces + boundary + NNC), identical scheme to v1
isInt=N(:,1)>0 & N(:,2)>0; oI=N(isInt,1); nI=N(isInt,2); FI=state.flux(isInt);
bf=find(~isInt); ownB=max(N(bf,1),N(bf,2)); sgnB=ones(numel(bf),1); sgnB(N(bf,1)==0)=-1; FB=state.flux(bf).*sgnB;
xbf=G.faces.centroids(bf,1); inSB=zeros(numel(bf),1); inSB(abs(xbf-1)<1e-9)=RIGHT_BC_S;
owner =[oI; ownB;               mfMatCell];
neigh =[nI; -ones(numel(bf),1); mfFracCell];
Fout  =[FI; FB;                 m2f];
inletS=[zeros(nnz(isInt),1); inSB; zeros(numel(m2f),1)];
hasNb=neigh>0;
outflux=accumarray(owner,max(Fout,0),[ncell,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[ncell,1]);
act=outflux>1e-30; dt_cfl=CFL*min(pv(act)./(FPRIME_MAX*outflux(act)));
fbl=@(s) s.^2./(s.^2+(1-s).^2+1e-30); nConn=numel(Fout);
outfl=Fout>=0; isBin=~outfl&~hasNb; upCell=ones(nConn,1);
upCell(outfl)=owner(outfl); sel=~outfl&hasNb; upCell(sel)=neigh(sel);
w_bin=fbl(inletS).*Fout; hb=find(hasNb);
Inc=sparse([owner; neigh(hb)],[(1:nConn)'; hb],[ones(nConn,1); -ones(numel(hb),1)],ncell,nConn);

%% 10 evenly-spaced snapshots up to PVI=1.0
snap_PVI=(1:10)/10; snap_T_abs=snap_PVI*T_final;   % T_final = PVI=1 absolute time
sw_matrix_snaps=zeros(nc,10); sw_frac_snaps=zeros(nfrac,10);
fprintf('transport: dt_cfl=%.3e, %d snapshots to PVI=1.0 (T=%.5f)\n',dt_cfl,10,T_final);
S=zeros(ncell,1); tprev=0; tS=tic;
for sn=1:10
  Ttar=snap_T_abs(sn); ns=max(1,ceil((Ttar-tprev)/dt_cfl)); dt=(Ttar-tprev)/ns; dt_pv=dt./pv;
  for it=1:ns
    w=fbl(S(upCell)).*Fout; w(isBin)=w_bin(isBin);
    S=min(max(S-dt_pv.*(Inc*w),0),1);
  end
  sw_matrix_snaps(:,sn)=S(1:nc); sw_frac_snaps(:,sn)=S(nc+1:end); tprev=Ttar;
  fprintf('  snap %2d/10  PVI=%.1f  T=%.5f  (%d steps)  Sw_max=%.4f  Sfrac_max=%.4f  [%.1fs]\n', ...
          sn,snap_PVI(sn),Ttar,ns,max(S(1:nc)),max(S(nc+1:end)),toc(tS));
end
%% PVI=1.0 snapshot -> v1-compatible single-snapshot fields
sw_matrix=sw_matrix_snaps(:,10); sw_frac=sw_frac_snaps(:,10);
sw_matrix_matched=sw_matrix; sw_frac_matched=sw_frac; s_matrix=sw_matrix; s_matrix_matched=sw_matrix;
%% absolute-time conversion for the notebook's stop rule
meta_PVI_to_Tabs=PV_matrix/Qright; meta_snap_PVI=snap_PVI; meta_snap_T_abs=snap_T_abs;
meta_transport_solver='explicit first-order upwind on MRST/EDFM grid with frozen MRST fluxes (matrix faces + NNC m2f); F(S)=S^2/(S^2+(1-S)^2); S=1 on x=1; sealed fracture tips';
if ~exist('meta_mrst_version','var')||isempty(meta_mrst_version)||~ischar(meta_mrst_version), meta_mrst_version='2024b'; end
meta_mrst_version='MRST 2024b (SINTEF, per startup banner; install folder mrst-2025a)';

%% ===== full v2 export (all v1 field names + 10 snapshots + times + conventions + provenance) =====
save(outfile,'-v7', ...
  'p_matrix','xc_matrix','p_frac','xc_frac','s_frac','s_frac_arc', ...
  'face_p1','face_p2','face_centroid','face_normal','face_len','face_flux', ...
  'face_neighbors','face_is_boundary','face_frac_cut','frac_face_flux','frac_face_neighbors', ...
  'lam_s','lam_xy','lam_seglen','lam_flux','lam_density', ...
  'nnc_mat_cell','nnc_frac_cell','nnc_flux_m2f','nnc_s', ...
  'sw_frac','sw_matrix','sw_frac_matched','sw_matrix_matched','s_matrix','s_matrix_matched', ...
  'sw_matrix_snaps','sw_frac_snaps','snap_PVI','snap_T_abs','meta_snap_PVI','meta_snap_T_abs','meta_PVI_to_Tabs', ...
  'tip_cells','tip_flux','tip_xy','conventions','meta_mrst_version','meta_modules','meta_mrst_root', ...
  'meta_check_tipface_flux','meta_check_bc_on_frac','meta_check_n_tipfaces','meta_check_massbal_residual','meta_check_net_nnc', ...
  'meta_celldim','meta_physdim','meta_fracA','meta_fracB','meta_tau','meta_aperture','meta_kf', ...
  'meta_km','meta_Kgamma','meta_nc','meta_nfrac','meta_phi_matrix','meta_phi_fracture','meta_Q_water', ...
  'meta_PV_matrix','meta_PV_frac','meta_PV_total','meta_CFL','meta_FPRIME_MAX','meta_T_final_PVI1', ...
  'meta_tip_bc','meta_transport_solver');
fprintf('saved %s\n',outfile);
fprintf('snapshots: matrix Sw range [%.4f %.4f], fracture Sw range [%.4f %.4f] at PVI=1.0\n', ...
        min(sw_matrix),max(sw_matrix),min(sw_frac),max(sw_frac));
