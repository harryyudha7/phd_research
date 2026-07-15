%% Transport on the heterogeneous tanh-perm field (Koppel-Martin BCs, no fracture).
%% Flow x=1(p=4) -> x=0(p=1); inflow S=1 at x=1; no-flow top/bottom; f=0.
%% Two snapshots PVI=0.3 and 0.5. Hand-code + MRST explicitTransport (independent).
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
gravity reset off
Khi=10; delta=0.1; CFL=0.45; FPRIME_MAX=2.0;   % more diffuse interface
kfun=@(XY,d) 1 + (Khi-1)*0.5.*(1+tanh((XY(:,1)+XY(:,2)-1)/d));

G=computeGeometry(cartGrid([128 128],[1 1])); xc=G.cells.centroids; N=G.faces.neighbors;
rock.perm=kfun(xc,delta); rock.poro=ones(G.cells.num,1); G.rock=rock;
hT=computeTrans(G,rock); T_harm=1./accumarray(G.cells.faces(:,1),1./hT,[G.faces.num,1]);
bc=pside(pside([],G,'LEFT',1),G,'RIGHT',4);
state=incompTPFA(initResSol(G,0),G,T_harm,initSingleFluid('mu',1,'rho',1),'bc',bc,'use_trans',true);

bf0=find(any(N==0,2)); xb0=G.faces.centroids(bf0,1);
Qin=sum(abs(state.flux(bf0(abs(xb0-1)<1e-9))));            % inflow at x=1
pv=poreVolume(G,rock); PV=sum(pv);
T1=0.3*PV/Qin; T2=0.5*PV/Qin;
fprintf('\n=== tanh transport (Koppel-Martin BCs, no fracture) ===\n');
fprintf('Q_in(x=1)=%.5f PV=%.4f -> T(PVI=0.3)=%.5f  T(PVI=0.5)=%.5f\n',Qin,PV,T1,T2);

%% connection list (matrix faces only); inflow S=1 on x=1
isInt=N(:,1)>0&N(:,2)>0; oI=N(isInt,1); nI=N(isInt,2); FI=state.flux(isInt);
bf=find(~isInt); ownB=max(N(bf,1),N(bf,2)); sgnB=ones(numel(bf),1); sgnB(N(bf,1)==0)=-1;
FB=state.flux(bf).*sgnB; xbf=G.faces.centroids(bf,1);
inSB=zeros(numel(bf),1); inSB(abs(xbf-1)<1e-9)=1;
owner=[oI;ownB]; neigh=[nI;-ones(numel(bf),1)]; Fout=[FI;FB];
inletS=[zeros(nnz(isInt),1);inSB]; hasNb=neigh>0;
outflux=accumarray(owner,max(Fout,0),[G.cells.num,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[G.cells.num,1]);
act=outflux>1e-30; dt_cfl=CFL*min(pv(act)./(FPRIME_MAX*outflux(act)));
s_h_t1=march_up(T1,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,G.cells.num);
s_h_t2=march_up(T2,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,G.cells.num);

%% MRST explicitTransport (independent; f=0 so it runs)
fluid2=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
bc2=pside([],G,'LEFT',1,'sat',[0 1]); bc2=pside(bc2,G,'RIGHT',4,'sat',[1 0]);
st=incompTPFA(initResSol(G,0,[0 1]),G,T_harm,fluid2,'bc',bc2,'use_trans',true);
st=explicitTransport(st,G,T1,rock,fluid2,'bc',bc2,'Trans',T_harm); s_e_t1=st.s(:,1);
st=explicitTransport(st,G,T2-T1,rock,fluid2,'bc',bc2,'Trans',T_harm); s_e_t2=st.s(:,1);
rmse=@(a,b) sqrt(mean((a-b).^2));
fprintf('PVI=0.3: indep max=%.4f hand max=%.4f RMSE=%.3e  Sw range [%.4f %.4f]\n',max(s_e_t1),max(s_h_t1),rmse(s_e_t1,s_h_t1),min(s_e_t1),max(s_e_t1));
fprintf('PVI=0.5: indep max=%.4f hand max=%.4f RMSE=%.3e\n',max(s_e_t2),max(s_h_t2),rmse(s_e_t2,s_h_t2));

%% extend the tanh export with transport (one file, frozen flux)
S=load('c:\Users\muchamad\mrst-project\case3_mrst_export_tanh_d0p1.mat');
S.sw_matrix_pvi03=s_e_t1(:); S.sw_matrix_pvi05=s_e_t2(:);
S.sw_matrix_pvi03_matched=s_h_t1(:); S.sw_matrix_pvi05_matched=s_h_t2(:);
S.meta_Q_in=Qin; S.meta_PV=PV; S.meta_T_pvi03=T1; S.meta_T_pvi05=T2;
S.meta_inflow='S=1 inflow at x=1; outflow x=0; no-flow top/bottom; f=0';
S.meta_transport_solver='MRST explicitTransport (sw_*) ; hand-coded upwind (sw_*_matched)';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_tanh_d0p1.mat','-struct','S','-v7');
fprintf('updated case3_mrst_export_tanh_d0p1.mat with sw_matrix_pvi03/pvi05\n');

figure('Name','tanh Sw PVI=0.3','Position',[80 80 520 470]);
plotCellData(G,s_e_t1,'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]);
hold on; line([0 1],[1 0],'Color','k','LineStyle','--'); view(0,90); axis equal tight; colorbar;
title(sprintf('tanh S_w at PVI=0.3 (T=%.4f)',T1)); xlabel x; ylabel y;
figure('Name','tanh Sw PVI=0.5','Position',[620 80 520 470]);
plotCellData(G,s_e_t2,'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]);
hold on; line([0 1],[1 0],'Color','k','LineStyle','--'); view(0,90); axis equal tight; colorbar;
title(sprintf('tanh S_w at PVI=0.5 (T=%.4f)',T2)); xlabel x; ylabel y;

function S=march_up(Ttar,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,ncell)
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
