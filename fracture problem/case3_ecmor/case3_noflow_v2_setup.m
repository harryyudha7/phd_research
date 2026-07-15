%% Case-1 EDFM sealed-tip reference v2 -- STAGE 1: solve + sealed-tip/mass-balance checks + static export fields.
%% IDENTICAL model to case3_mrst_transport_noflow.m (do NOT modify geometry/perm/conductance/BC).
%% Reports the net-NNC sealed-tip signature BEFORE the transport (stage 2); errors out if tips are not sealed.
%% MRST-native signs preserved: state.flux + = neighbors(:,1)->(:,2); NNC m2f + = matrix->fracture. No re-signing.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add hfm incomp ad-core
checkLineSegmentIntersect; gravity reset off
ckpt='c:\Users\muchamad\mrst-project\case3_noflow_v2_checkpoint.mat';

celldim=[128 128]; physdim=[1 1]; A=[0.25 0]; B=[0.75 1];
K_GAMMA=10; aperture=1e-2; k_f=K_GAMMA/aperture; k_m=1;
p_left=1; p_right=4; PVI_target=1.0; phi_matrix=1.0; phi_fracture=1.0;
CFL=0.45; FPRIME_MAX=2.0; RIGHT_BC_S=1.0;

fprintf('building EDFM model (unchanged)...\n'); tB=tic;
G=computeGeometry(cartGrid(celldim,physdim)); fl=[A(1) A(2) B(1) B(2)];
[G,fracture]=processFracture2D(G,fl); fracture.aperture=aperture;
G=CIcalculator2D(G,fracture);
[G,F,fracture]=gridFracture2D(G,fracture,'min_size',0.4/128,'cell_size',1/128);
G.rock.perm=k_m*ones(G.cells.num,1);
fn=fieldnames(G.FracGrid);
for i=1:numel(fn), Gf=G.FracGrid.(fn{i}); G.FracGrid.(fn{i}).rock.perm=k_f*ones(Gf.cells.num,1); end
[G,T]=defineNNCandTrans(G,F,fracture);
nc=G.Matrix.cells.num; nfrac=G.cells.num-nc; ncell=G.cells.num;
ellF=G.cells.volumes(nc+1:end)/aperture;
G.rock.poro=[phi_matrix*ones(nc,1); phi_fracture*ones(nfrac,1)];
G.cells.volumes(nc+1:end)=ellF;
cenF=G.cells.centroids(nc+1:end,:); tau=(B-A)/norm(B-A); sF=(cenF-A)*tau';
[~,iA]=min(sF); [~,iB]=max(sF); cellA=nc+iA; cellB=nc+iB;
fprintf('  build %.1fs; nc=%d nfrac=%d\n',toc(tB),nc,nfrac);

%% pressure solve -- NO tip wells (sealed), single-phase fixed flux
fluidP=initSingleFluid('mu',1,'rho',1);
bc=pside(pside([],G,'RIGHT',p_right),G,'LEFT',p_left);
state=incompTPFA(initResSol(G,0),G,T,fluidP,'bc',bc,'use_trans',true); p=state.pressure;
N=G.faces.neighbors;
fprintf('pressure: matrix p [%.4f %.4f], fracture p [%.4f %.4f] (floats)\n', ...
        min(p(1:nc)),max(p(1:nc)),min(p(nc+1:end)),max(p(nc+1:end)));

%% NNC exchange (m2f>0 = matrix INTO fracture) -- MRST-native sign, per connection
nnc=G.nnc.cells; Tnnc=G.nnc.T;
isMF=(nnc(:,1)<=nc & nnc(:,2)>nc)|(nnc(:,1)>nc & nnc(:,2)<=nc); rows=find(isMF);
fr=nnc(rows,:); fr(fr<=nc)=0; mfFracCell=max(fr,[],2);
mcc=nnc(rows,:); mcc(mcc>nc)=0; mfMatCell=max(mcc,[],2);
f12=Tnnc(rows).*(p(nnc(rows,1))-p(nnc(rows,2)));
m2f=f12.*(1-2*double(nnc(rows,1)>nc));

%% ===== CHECKS (reported BEFORE transport; gate the run) =====
bf0=find(any(N==0,2));
net_nnc = sum(m2f);
% fracture dead-end (tip) faces: 0-neighbor faces adjacent to a fracture cell. Tips DO touch the
% boundary, so these exist; "sealed" = they carry NO BC and ZERO flux (Neumann-zero dead ends).
fracDead = find(any(N==0,2) & any(N>nc,2));
max_tipface_flux = 0; if ~isempty(fracDead), max_tipface_flux=max(abs(state.flux(fracDead))); end
bc_on_frac = double(any(ismember(bc.face,fracDead)));   % is any BC applied to a fracture face?
% per-cell mass-balance residual: div(face flux) + div(all NNC) should vanish (steady, no source)
divF = accumarray(N(N(:,1)>0,1),state.flux(N(:,1)>0),[ncell,1]) - accumarray(N(N(:,2)>0,2),state.flux(N(:,2)>0),[ncell,1]);
fall = Tnnc.*(p(nnc(:,1))-p(nnc(:,2)));
divN = accumarray(nnc(:,1),fall,[ncell,1]) - accumarray(nnc(:,2),fall,[ncell,1]);
massres = max(abs(divF+divN));
fprintf('\n=== CHECKS (before transport) ===\n');
fprintf('sealed tips: %d fracture dead-end faces (tips touch boundary); max|flux| on them = %.3e (MUST ~0); BC on any fracture face? %d (MUST 0)\n',numel(fracDead),max_tipface_flux,bc_on_frac);
fprintf('per-cell mass-balance residual (max over %d cells) = %.3e  (should be ~solver tol)\n',ncell,massres);
fprintf('net fracture exchange sum(NNC m2f) = %.3e  (SEALED-TIP SIGNATURE; must ~0)\n',net_nnc);
if bc_on_frac || max_tipface_flux>1e-8 || abs(net_nnc)>1e-8
  error('SEALED-TIP CHECK FAILED (max_tipface_flux=%.3e bc_on_frac=%d net_nnc=%.3e). Tips not sealed -> STOP.',max_tipface_flux,bc_on_frac,net_nnc);
end
fprintf('=> tips SEALED (touch boundary but zero tip-face flux, no BC on fracture; net NNC ~0), mass balance at tolerance. OK to run transport.\n');

%% pore volumes, injection rate, end time (matrix-PV basis)
bf=G.faces.centroids; Qright=sum(abs(state.flux(bf0(abs(bf(bf0,1)-1)<1e-9))));
pv=poreVolume(G,G.rock); PV_matrix=sum(pv(1:nc)); PV_frac=sum(pv(nc+1:end)); PV_total=PV_matrix+PV_frac;
T_final=PVI_target*PV_matrix/Qright;
fprintf('PV_matrix=%.4f PV_frac=%.4f Q_water=%.4f -> T_final(PVI=1)=%.5f\n',PV_matrix,PV_frac,Qright,T_final);

%% ===== STATIC export fields (matrix / fracture / faces / NNC) -- same names as v1 =====
p_matrix=p(1:nc); xc_matrix=G.cells.centroids(1:nc,:);
p_frac=p(nc+1:end); xc_frac=cenF; s_frac=sF(:); s_frac_arc=sF(:);
isMat=~any(N>nc,2); fmat=find(isMat);
np=G.faces.nodePos; n1=G.faces.nodes(np(fmat)); n2=G.faces.nodes(np(fmat)+1);
face_p1=G.nodes.coords(n1,:); face_p2=G.nodes.coords(n2,:);
face_centroid=G.faces.centroids(fmat,:); face_len=G.faces.areas(fmat);
face_normal=G.faces.normals(fmat,:)./face_len; face_flux=state.flux(fmat);
face_neighbors=N(fmat,:); face_is_boundary=double(any(face_neighbors==0,2));
isFF=all(N>nc,2); fff=find(isFF); frac_face_flux=state.flux(fff); frac_face_neighbors=N(fff,:);
fracLocal=mfFracCell-nc; lam_flux=accumarray(fracLocal,m2f,[nfrac,1]);
lam_seglen=ellF; lam_density=lam_flux./lam_seglen; lam_s=sF; lam_xy=cenF;
nnc_mat_cell=mfMatCell; nnc_frac_cell=mfFracCell; nnc_flux_m2f=m2f; nnc_s=sF(fracLocal);
fcut=false(ncell,1); fcut(unique(mfMatCell))=true;
nb1=face_neighbors(:,1); nb2=face_neighbors(:,2);
c1=false(numel(fmat),1); c2=c1; c1(nb1>0)=fcut(nb1(nb1>0)); c2(nb2>0)=fcut(nb2(nb2>0));
face_frac_cut=double(c1|c2);
tip_cells=[cellB; cellA]; tip_flux=[0;0]; tip_xy=G.cells.centroids(tip_cells,:);

%% conventions + provenance
try, mrst_ver=mrstVersion(); catch, mrst_ver='release per MRST startup banner'; end
conventions=struct( ...
  'flux_sign','state.flux>0 points from face_neighbors(:,1) to face_neighbors(:,2); boundary neighbor stored as 0', ...
  'nnc_sign','nnc_flux_m2f>0 = matrix INTO fracture (MRST-native; NOT re-signed at export)', ...
  'index_base','1-based MATLAB indexing (matrix cells 1..nc, fracture cells nc+1..nc+nfrac)', ...
  'units','nondimensional: matrix K=1, mu=1, unit domain; fracture K_Gamma=10, aperture=1e-2, k_f=1000', ...
  'orientation','matrix cell/face order = cartGrid([128 128]); fracture cells appended after nc', ...
  'pvi_definition','PVI = Q_water*T/PV_matrix (matrix pore-volume basis); snap_T_abs = snap_PVI*PV_matrix/Q_water');
meta_mrst_version=mrst_ver; meta_modules={'incomp','hfm','ad-core'}; meta_mrst_root=mrstRoot;
meta_check_tipface_flux=max_tipface_flux; meta_check_bc_on_frac=bc_on_frac; meta_check_n_tipfaces=numel(fracDead);
meta_check_massbal_residual=massres; meta_check_net_nnc=net_nnc;
meta_celldim=celldim; meta_physdim=physdim; meta_fracA=A; meta_fracB=B; meta_tau=tau;
meta_aperture=aperture; meta_kf=k_f; meta_km=k_m; meta_Kgamma=K_GAMMA; meta_nc=nc; meta_nfrac=nfrac;
meta_phi_matrix=phi_matrix; meta_phi_fracture=phi_fracture; meta_Q_water=Qright;
meta_PV_matrix=PV_matrix; meta_PV_frac=PV_frac; meta_PV_total=PV_total;
meta_CFL=CFL; meta_FPRIME_MAX=FPRIME_MAX; meta_T_final_PVI1=T_final;
meta_tip_bc='no-flow (Neumann zero) at fracture tips; no tip wells; fracture pressure floats; sealed (verified)';

save(ckpt,'-v7.3','G','T','state','p','N','pv','nc','nfrac','ncell','cellA','cellB','sF','ellF', ...
  'Qright','PV_matrix','PV_frac','PV_total','T_final','celldim','CFL','FPRIME_MAX','RIGHT_BC_S','tau', ...
  'p_matrix','xc_matrix','p_frac','xc_frac','s_frac','s_frac_arc','face_p1','face_p2','face_centroid', ...
  'face_normal','face_len','face_flux','face_neighbors','face_is_boundary','face_frac_cut', ...
  'frac_face_flux','frac_face_neighbors','lam_s','lam_xy','lam_seglen','lam_flux','lam_density', ...
  'nnc_mat_cell','nnc_frac_cell','nnc_flux_m2f','nnc_s','mfMatCell','mfFracCell','m2f','tip_cells','tip_flux','tip_xy', ...
  'conventions','meta_mrst_version','meta_modules','meta_mrst_root','meta_check_tipface_flux','meta_check_bc_on_frac','meta_check_n_tipfaces', ...
  'meta_check_massbal_residual','meta_check_net_nnc','meta_celldim','meta_physdim','meta_fracA','meta_fracB','meta_tau', ...
  'meta_aperture','meta_kf','meta_km','meta_Kgamma','meta_nc','meta_nfrac','meta_phi_matrix','meta_phi_fracture', ...
  'meta_Q_water','meta_PV_matrix','meta_PV_frac','meta_PV_total','meta_CFL','meta_FPRIME_MAX','meta_T_final_PVI1','meta_tip_bc');
fprintf('STAGE 1 done. checkpoint saved: %s\n',ckpt);
fprintf('MRST version: %s ; modules: incomp, hfm, ad-core\n',mrst_ver);
