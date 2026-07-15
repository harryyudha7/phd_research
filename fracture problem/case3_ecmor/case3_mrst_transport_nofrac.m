%% NO-FRACTURE control case. Same matrix problem, NO fracture at all.
%% One frozen single-phase flux; transport reported at TWO final times:
%%   T1 = PVI=1 (= PV_matrix/Q_water = 1/3),  T2 = 0.35 (rounded, PVI=1.05).
%% Hand-coded explicit upwind + independent MRST explicitTransport. One export file.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
gravity reset off

celldim=[128 128]; physdim=[1 1]; k_m=1; phi_matrix=1.0;
p_left=1; p_right=4; CFL=0.45; FPRIME_MAX=2.0; RIGHT_BC_S=1.0;
T2=0.35;                       % rounded second time (user choice)

%% grid + rock (no fracture) + pressure solve
G=computeGeometry(cartGrid(celldim,physdim));
rock=makeRock(G,k_m,phi_matrix); G.rock=rock;
Tr=computeTrans(G,rock);
fluidP=initSingleFluid('mu',1,'rho',1);
bc=pside(pside([],G,'RIGHT',p_right),G,'LEFT',p_left);
state=incompTPFA(initResSol(G,0),G,Tr,fluidP,'bc',bc);
p=state.pressure; N=G.faces.neighbors;
bf0=find(any(N==0,2)); xb0=G.faces.centroids(bf0,1);
Qright=sum(abs(state.flux(bf0(abs(xb0-1)<1e-9))));
pv=poreVolume(G,rock); PV_matrix=sum(pv);
T1=1.0*PV_matrix/Qright;       % PVI = 1
PVI2=Qright*T2/PV_matrix;
fprintf('\n=== no-fracture pressure solve ===\n');
fprintf('matrix p [%.4f %.4f], Q_water=%.5f, PV_matrix=%.4f\n',min(p),max(p),Qright,PV_matrix);
fprintf('T1(PVI=1)=%.5f   T2=%.5f (PVI=%.4f)\n',T1,T2,PVI2);

%% connection list (matrix faces only)
ncell=G.cells.num;
isInt=N(:,1)>0 & N(:,2)>0; oI=N(isInt,1); nI=N(isInt,2); FI=state.flux(isInt);
bf=find(~isInt); ownB=max(N(bf,1),N(bf,2));
sgnB=ones(numel(bf),1); sgnB(N(bf,1)==0)=-1; FB=state.flux(bf).*sgnB;
xbf=G.faces.centroids(bf,1); inSB=zeros(numel(bf),1); inSB(abs(xbf-1)<1e-9)=RIGHT_BC_S;
owner=[oI;ownB]; neigh=[nI;-ones(numel(bf),1)]; Fout=[FI;FB];
inletS=[zeros(nnz(isInt),1);inSB]; hasNb=neigh>0;
outflux=accumarray(owner,max(Fout,0),[ncell,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[ncell,1]);
act=outflux>1e-30; dt_cfl=CFL*min(pv(act)./(FPRIME_MAX*outflux(act)));

%% hand-coded explicit upwind at both times
s_h_t1=march_upwind(T1,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,ncell);
s_h_t2=march_upwind(T2,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,ncell);

%% independent MRST explicitTransport at both times (continue T1 -> T2)
fluid2=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
bc2=pside([],G,'RIGHT',p_right,'sat',[1 0]); bc2=pside(bc2,G,'LEFT',p_left,'sat',[0 1]);
st=incompTPFA(initResSol(G,0,[0 1]),G,Tr,fluid2,'bc',bc2); tE=tic;
st=explicitTransport(st,G,T1,rock,fluid2,'bc',bc2,'Trans',Tr); s_e_t1=st.s(:,1);
st=explicitTransport(st,G,T2-T1,rock,fluid2,'bc',bc2,'Trans',Tr); s_e_t2=st.s(:,1);
fprintf('explicitTransport (both times) %.1fs\n',toc(tE));
rmse=@(a,b) sqrt(mean((a-b).^2));
fprintf('\n=== no-fracture: explicitTransport vs hand-code ===\n');
fprintf('T1: indep max=%.4f hand max=%.4f RMSE=%.3e\n',max(s_e_t1),max(s_h_t1),rmse(s_e_t1,s_h_t1));
fprintf('T2: indep max=%.4f hand max=%.4f RMSE=%.3e\n',max(s_e_t2),max(s_h_t2),rmse(s_e_t2,s_h_t2));

%% ===== EXPORT (one file; flux frozen, two saturation snapshots) =====
p_matrix=p; xc_matrix=G.cells.centroids;
np=G.faces.nodePos; allF=(1:G.faces.num)';
n1=G.faces.nodes(np(allF)); n2=G.faces.nodes(np(allF)+1);
face_p1=G.nodes.coords(n1,:); face_p2=G.nodes.coords(n2,:);
face_centroid=G.faces.centroids; face_len=G.faces.areas;
face_normal=G.faces.normals./face_len; face_flux=state.flux;
face_neighbors=N; face_is_boundary=double(any(N==0,2));
sw_matrix_t1=s_e_t1(:); sw_matrix_t2=s_e_t2(:);                   % MRST explicitTransport
sw_matrix_t1_matched=s_h_t1(:); sw_matrix_t2_matched=s_h_t2(:);  % hand-code
meta_celldim=celldim; meta_physdim=physdim; meta_km=k_m; meta_phi_matrix=phi_matrix;
meta_Q_water=Qright; meta_PV_matrix=PV_matrix;
meta_T1=T1; meta_T2=T2; meta_PVI1=1.0; meta_PVI2=PVI2;
meta_CFL=CFL; meta_FPRIME_MAX=FPRIME_MAX; meta_nc=ncell;
meta_tip_bc='none (no fracture in this case)';
meta_transport_solver='MRST explicitTransport (independent) for sw_*; hand-coded upwind for sw_*_matched';
README=['NO-FRACTURE control: pure matrix BL transport, frozen single-phase flux. ', ...
        'sw_matrix_t1 = MRST explicitTransport at T1=PVI1 (=PV/Q=1/3). ', ...
        'sw_matrix_t2 = same at T2 (rounded). _matched = hand-coded upwind. No fracture/lambda/tips.'];
save('c:\Users\muchamad\mrst-project\case3_mrst_export_nofrac.mat','-v7', ...
   'p_matrix','xc_matrix','face_p1','face_p2','face_centroid','face_normal','face_len', ...
   'face_flux','face_neighbors','face_is_boundary', ...
   'sw_matrix_t1','sw_matrix_t2','sw_matrix_t1_matched','sw_matrix_t2_matched', ...
   'meta_celldim','meta_physdim','meta_km','meta_phi_matrix','meta_Q_water','meta_PV_matrix', ...
   'meta_T1','meta_T2','meta_PVI1','meta_PVI2','meta_CFL','meta_FPRIME_MAX','meta_nc', ...
   'meta_tip_bc','meta_transport_solver','README');
fprintf('\nsaved case3_mrst_export_nofrac.mat  (T1=%.5f PVI=1 ; T2=%.5f PVI=%.3f)\n',T1,T2,PVI2);

%% ---- local function: hand-coded explicit first-order upwind to a target time ----
function S=march_upwind(Ttar,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,ncell)
  fbl=@(s)s.^2./(s.^2+(1-s).^2+1e-30);
  nsteps=max(1,ceil(Ttar/dt_cfl)); dt=Ttar/nsteps; nConn=numel(Fout);
  outfl=Fout>=0; isBin=~outfl&~hasNb; upCell=ones(nConn,1);
  upCell(outfl)=owner(outfl); sel=~outfl&hasNb; upCell(sel)=neigh(sel);
  w_bin=fbl(inletS).*Fout; hb=find(hasNb);
  Inc=sparse([owner;neigh(hb)],[(1:nConn)';hb],[ones(nConn,1);-ones(numel(hb),1)],ncell,nConn);
  dt_pv=dt./pv; S=zeros(ncell,1);
  for it=1:nsteps
    w=fbl(S(upCell)).*Fout; w(isBin)=w_bin(isBin);
    S=min(max(S-dt_pv.*(Inc*w),0),1);
  end
  fprintf('  hand-code march to T=%.5f: %d steps, Sw_max=%.4f\n',Ttar,nsteps,max(S));
end
