%% Case 3 -- NO-FLOW fracture tips (Neumann zero) variant.
%% Identical to case3_mrst_transport.m EXCEPT the fracture tip BC: no tip wells,
%% so the fracture ends are sealed (no throughflow, fracture pressure floats).
%% Hand-coded explicit first-order upwind (same scheme); single-phase fixed flux;
%% T_final recomputed from the new Q_water to keep PVI=1.2.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add hfm incomp ad-core
checkLineSegmentIntersect; gravity reset off

celldim=[128 128]; physdim=[1 1]; A=[0.25 0]; B=[0.75 1];
K_GAMMA=10; aperture=1e-2; k_f=K_GAMMA/aperture; k_m=1;
p_left=1; p_right=4; PVI_target=1.0; phi_matrix=1.0; phi_fracture=1.0;
CFL=0.45; FPRIME_MAX=2.0; RIGHT_BC_S=1.0;

fprintf('building EDFM model...\n'); tB=tic;
G=computeGeometry(cartGrid(celldim,physdim)); fl=[A(1) A(2) B(1) B(2)];
[G,fracture]=processFracture2D(G,fl); fracture.aperture=aperture;
G=CIcalculator2D(G,fracture);
[G,F,fracture]=gridFracture2D(G,fracture,'min_size',0.4/128,'cell_size',1/128);
G.rock.perm=k_m*ones(G.cells.num,1);
fn=fieldnames(G.FracGrid);
for i=1:numel(fn), Gf=G.FracGrid.(fn{i}); G.FracGrid.(fn{i}).rock.perm=k_f*ones(Gf.cells.num,1); end
[G,T]=defineNNCandTrans(G,F,fracture);
fprintf('  build %.1fs\n',toc(tB));
nc=G.Matrix.cells.num; nfrac=G.cells.num-nc;
ellF=G.cells.volumes(nc+1:end)/aperture;
G.rock.poro=[phi_matrix*ones(nc,1); phi_fracture*ones(nfrac,1)];
G.cells.volumes(nc+1:end)=ellF;

cenF=G.cells.centroids(nc+1:end,:); tau=(B-A)/norm(B-A); sF=(cenF-A)*tau';
[~,iA]=min(sF); [~,iB]=max(sF); cellA=nc+iA; cellB=nc+iB;   % tip cells (now no-flow)

%% pressure solve -- NO tip wells (fracture tips no-flow) ; single-phase fixed flux
fluidP=initSingleFluid('mu',1,'rho',1);
bc=pside(pside([],G,'RIGHT',p_right),G,'LEFT',p_left);
state=incompTPFA(initResSol(G,0),G,T,fluidP,'bc',bc,'use_trans',true);
p=state.pressure;
fprintf('\n=== pressure solve (no-flow tips) ===\n');
fprintf('matrix p [%.4f %.4f], fracture p [%.4f %.4f] (floats; no tip pinning)\n', ...
        min(p(1:nc)),max(p(1:nc)),min(p(nc+1:end)),max(p(nc+1:end)));

%% pore volumes, water-injection rate, end time (matrix PV=1 basis), recomputed
N=G.faces.neighbors; bf0=find(any(N==0,2)); xb0=G.faces.centroids(bf0,1);
Qright=sum(abs(state.flux(bf0(abs(xb0-1)<1e-9))));
pv=poreVolume(G,G.rock);
PV_matrix=sum(pv(1:nc)); PV_frac=sum(pv(nc+1:end)); PV_total=PV_matrix+PV_frac;
T_final=PVI_target*PV_matrix/Qright;        % recomputed (NOT the Dirichlet 0.47938)
fprintf('PV_matrix=%.4f PV_frac=%.4f Q_water=%.4f -> T_final=%.5f (PVI=%.1f)\n', ...
        PV_matrix,PV_frac,Qright,T_final,PVI_target);

%% matrix<->fracture NNC exchange (m2f>0 = matrix INTO fracture)
nnc=G.nnc.cells; Tnnc=G.nnc.T;
isMF=(nnc(:,1)<=nc & nnc(:,2)>nc)|(nnc(:,1)>nc & nnc(:,2)<=nc); rows=find(isMF);
fr=nnc(rows,:); fr(fr<=nc)=0; mfFracCell=max(fr,[],2);
mcc=nnc(rows,:); mcc(mcc>nc)=0; mfMatCell=max(mcc,[],2);
f12=Tnnc(rows).*(p(nnc(rows,1))-p(nnc(rows,2)));
m2f=f12.*(1-2*double(nnc(rows,1)>nc));

%% EXPLICIT FIRST-ORDER UPWIND (unified connection list, NO tips)
ncell=G.cells.num;
isInt=N(:,1)>0 & N(:,2)>0; oI=N(isInt,1); nI=N(isInt,2); FI=state.flux(isInt);
bf=find(~isInt); ownB=max(N(bf,1),N(bf,2));
sgnB=ones(numel(bf),1); sgnB(N(bf,1)==0)=-1; FB=state.flux(bf).*sgnB;
xbf=G.faces.centroids(bf,1); inSB=zeros(numel(bf),1); inSB(abs(xbf-1)<1e-9)=RIGHT_BC_S;
owner =[oI; ownB;               mfMatCell];
neigh =[nI; -ones(numel(bf),1); mfFracCell];
Fout  =[FI; FB;                 m2f];
inletS=[zeros(nnz(isInt),1); inSB; zeros(numel(m2f),1)];
hasNb=neigh>0;
outflux=accumarray(owner,max(Fout,0),[ncell,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[ncell,1]);
act=outflux>1e-30;
dt_cfl=CFL*min(pv(act)./(FPRIME_MAX*outflux(act)));
nsteps=max(1,ceil(T_final/dt_cfl)); dt=T_final/nsteps;
fprintf('explicit upwind: dt=%.4e nsteps=%d (CFL=%.2f)\n',dt,nsteps,CFL);

fbl=@(s) s.^2./(s.^2+(1-s).^2+1e-30); nConn=numel(Fout);
outfl=Fout>=0; isBin=~outfl&~hasNb; upCell=ones(nConn,1);
upCell(outfl)=owner(outfl); sel=~outfl&hasNb; upCell(sel)=neigh(sel);
w_bin=fbl(inletS).*Fout; hb=find(hasNb);
Inc=sparse([owner; neigh(hb)],[(1:nConn)'; hb],[ones(nConn,1); -ones(numel(hb),1)],ncell,nConn);
dt_pv=dt./pv; S=zeros(ncell,1); prog=max(1,round(nsteps/10)); tS=tic;
for it=1:nsteps
    w=fbl(S(upCell)).*Fout; w(isBin)=w_bin(isBin);
    S=min(max(S-dt_pv.*(Inc*w),0),1);
    if mod(it,prog)==0||it==nsteps
        fprintf('  %6d/%d (%3.0f%%) Sw_max=%.4f Sfrac_max=%.4f [%.1fs]\n',it,nsteps,100*it/nsteps,max(S(1:nc)),max(S(nc+1:end)),toc(tS));
    end
end
s_matrix_h=S(1:nc); s_frac_h=S(nc+1:end);
fprintf('=== PVI=%.2f (hand-code) === matrix Sw[%.4f %.4f] fracture Sw[%.4f %.4f]\n', ...
        PVI_target,min(s_matrix_h),max(s_matrix_h),min(s_frac_h),max(s_frac_h));

%% save two-phase setup for the independent explicitTransport run (no wells)
fluid2=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
bc2=pside([],G,'RIGHT',p_right,'sat',[1 0]); bc2=pside(bc2,G,'LEFT',p_left,'sat',[0 1]);
state2=incompTPFA(initResSol(G,0,[0 1]),G,T,fluid2,'bc',bc2,'use_trans',true);
save('c:\Users\muchamad\mrst-project\case3_noflow_setup.mat', ...
     'G','T','fluid2','bc2','state2','nc','nfrac','sF','T_final','-v7.3');

%% ===== EXPORT (same structure/naming as Dirichlet case) =====
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
tip_cells=[cellB; cellA]; tip_flux=[0;0]; tip_xy=G.cells.centroids(tip_cells,:);  % NO-FLOW tips
% saturations: hand-code now under *_matched and (placeholder) sw_*; explicitTransport overwrites sw_* next
sw_frac_matched=s_frac_h(:); sw_matrix_matched=s_matrix_h(:);
sw_frac=s_frac_h(:); sw_matrix=s_matrix_h(:); s_matrix=s_matrix_h(:); s_matrix_matched=s_matrix_h(:);
meta_celldim=celldim; meta_physdim=physdim; meta_fracA=A; meta_fracB=B; meta_tau=tau;
meta_aperture=aperture; meta_kf=k_f; meta_km=k_m; meta_Kgamma=K_GAMMA; meta_nc=nc; meta_nfrac=nfrac;
meta_PVI=PVI_target; meta_T_final=T_final; meta_total_injected=Qright*T_final;
meta_phi_matrix=phi_matrix; meta_phi_fracture=phi_fracture; meta_Q_water=Qright;
meta_PV_matrix=PV_matrix; meta_PV_frac=PV_frac; meta_PV_total=PV_total;
meta_CFL=CFL; meta_FPRIME_MAX=FPRIME_MAX; meta_dt=dt; meta_nsteps=nsteps;
meta_tip_bc='no-flow (Neumann zero) at fracture tips; no tip wells; fracture pressure floats';
meta_transport_solver='hand-coded explicit upwind (placeholder; explicitTransport overwrites sw_*)';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_noflow.mat','-v7', ...
   'p_matrix','xc_matrix','p_frac','xc_frac','s_frac','s_frac_arc', ...
   'face_p1','face_p2','face_centroid','face_normal','face_len','face_flux', ...
   'face_neighbors','face_is_boundary','face_frac_cut','frac_face_flux','frac_face_neighbors', ...
   'lam_s','lam_xy','lam_seglen','lam_flux','lam_density', ...
   'nnc_mat_cell','nnc_frac_cell','nnc_flux_m2f','nnc_s', ...
   'sw_frac','sw_matrix','sw_frac_matched','sw_matrix_matched','s_matrix','s_matrix_matched', ...
   'tip_cells','tip_flux','tip_xy', ...
   'meta_celldim','meta_physdim','meta_fracA','meta_fracB','meta_tau','meta_aperture','meta_kf', ...
   'meta_km','meta_Kgamma','meta_nc','meta_nfrac','meta_PVI','meta_T_final','meta_total_injected', ...
   'meta_phi_matrix','meta_phi_fracture','meta_Q_water','meta_PV_matrix','meta_PV_frac','meta_PV_total', ...
   'meta_CFL','meta_FPRIME_MAX','meta_dt','meta_nsteps','meta_tip_bc','meta_transport_solver');
fprintf('saved case3_mrst_export_noflow.mat + case3_noflow_setup.mat\n');
