%% NO-FRACTURE: add EARLIER snapshots (PVI=0.5 and rounded T=0.20) to show the front.
%% ADDS fields to case3_mrst_export_nofrac.mat; does NOT overwrite the PVI=1 / T=0.35 runs.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
gravity reset off

celldim=[128 128]; physdim=[1 1]; k_m=1; phi_matrix=1.0;
p_left=1; p_right=4; CFL=0.45; FPRIME_MAX=2.0; RIGHT_BC_S=1.0;
T4=0.20;                                   % rounded companion time (PVI=0.6)

G=computeGeometry(cartGrid(celldim,physdim));
rock=makeRock(G,k_m,phi_matrix); G.rock=rock; Tr=computeTrans(G,rock);
fluidP=initSingleFluid('mu',1,'rho',1);
bc=pside(pside([],G,'RIGHT',p_right),G,'LEFT',p_left);
state=incompTPFA(initResSol(G,0),G,Tr,fluidP,'bc',bc);
N=G.faces.neighbors; bf0=find(any(N==0,2)); xb0=G.faces.centroids(bf0,1);
Qright=sum(abs(state.flux(bf0(abs(xb0-1)<1e-9)))); pv=poreVolume(G,rock); PV_matrix=sum(pv);
T3=0.5*PV_matrix/Qright;                   % PVI = 0.5
fprintf('Q_water=%.5f  T3(PVI=0.5)=%.5f  T4=%.5f (PVI=%.3f)\n',Qright,T3,T4,Qright*T4/PV_matrix);

ncell=G.cells.num;
isInt=N(:,1)>0 & N(:,2)>0; oI=N(isInt,1); nI=N(isInt,2); FI=state.flux(isInt);
bf=find(~isInt); ownB=max(N(bf,1),N(bf,2)); sgnB=ones(numel(bf),1); sgnB(N(bf,1)==0)=-1;
FB=state.flux(bf).*sgnB; xbf=G.faces.centroids(bf,1); inSB=zeros(numel(bf),1); inSB(abs(xbf-1)<1e-9)=RIGHT_BC_S;
owner=[oI;ownB]; neigh=[nI;-ones(numel(bf),1)]; Fout=[FI;FB]; inletS=[zeros(nnz(isInt),1);inSB]; hasNb=neigh>0;
outflux=accumarray(owner,max(Fout,0),[ncell,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[ncell,1]);
act=outflux>1e-30; dt_cfl=CFL*min(pv(act)./(FPRIME_MAX*outflux(act)));

s_h_t3=march_upwind(T3,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,ncell);
s_h_t4=march_upwind(T4,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,ncell);

fluid2=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
bc2=pside([],G,'RIGHT',p_right,'sat',[1 0]); bc2=pside(bc2,G,'LEFT',p_left,'sat',[0 1]);
st=incompTPFA(initResSol(G,0,[0 1]),G,Tr,fluid2,'bc',bc2);
st=explicitTransport(st,G,T3,rock,fluid2,'bc',bc2,'Trans',Tr); s_e_t3=st.s(:,1);
st=explicitTransport(st,G,T4-T3,rock,fluid2,'bc',bc2,'Trans',Tr); s_e_t4=st.s(:,1);
fprintf('PVI=0.5: Sw range [%.4f %.4f]  (front visible if min<<max)\n',min(s_e_t3),max(s_e_t3));

%% ---- ADD to the existing export (preserve PVI=1 / T=0.35 fields) ----
f='c:\Users\muchamad\mrst-project\case3_mrst_export_nofrac.mat'; S=load(f);
assert(isfield(S,'sw_matrix_t1') && isfield(S,'sw_matrix_t2'),'existing snapshots missing!');
S.sw_matrix_pvi05=s_e_t3(:);  S.sw_matrix_t020=s_e_t4(:);
S.sw_matrix_pvi05_matched=s_h_t3(:);  S.sw_matrix_t020_matched=s_h_t4(:);
S.meta_T_pvi05=T3;  S.meta_T_t020=T4;  S.meta_PVI_t020=Qright*T4/PV_matrix;
save(f,'-struct','S','-v7');
fprintf('\nADDED sw_matrix_pvi05 (T=%.5f) + sw_matrix_t020 (T=%.2f) to case3_mrst_export_nofrac.mat\n',T3,T4);
fprintf('existing fields intact: sw_matrix_t1=%d sw_matrix_t2=%d\n', ...
        isfield(S,'sw_matrix_t1'), isfield(S,'sw_matrix_t2'));

%% ---- plot the PVI=0.5 front ----
figure('Name','no-fracture Sw, PVI=0.5','Position',[100 100 560 500]);
plotCellData(G,s_e_t3,'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]);
view(0,90); axis equal tight; colorbar;
title(sprintf('No-fracture S_w at PVI=0.5 (T=%.4f)',T3)); xlabel x; ylabel y;

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
