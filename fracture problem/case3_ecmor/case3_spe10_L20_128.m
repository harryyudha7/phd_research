%% SPE10 layer 20 (Tarbert) RESAMPLED to 128x128, pressure + transport.
%% Koppel-Martin BCs (p=1 x=0, p=4 x=1, no-flow top/bottom, f=0, no fracture).
%% Harmonic averaging. S=1 inflow at x=1. Two snapshots PVI=0.3, 0.5.
%% Hand-code + MRST explicitTransport. Writes ONLY case3_mrst_export_spe10_L20_128.mat.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add spe10 incomp
gravity reset off
LAYER=20; CFL=0.45; FPRIME_MAX=2.0;

%% SPE10 layer 20 -> normalize -> resample 60x220 to 128x128 (nearest block)
rk=getSPE10rock(LAYER); Kx=rk.perm(:,1); kappa=Kx./exp(mean(log(Kx)));
Korig=reshape(kappa,[60 220]);                        % (i=x, j=y)
ispe=min(60,max(1,ceil(((1:128)-0.5)/128*60)));
jspe=min(220,max(1,ceil(((1:128)-0.5)/128*220)));
kappa_grid=Korig(ispe,jspe);                          % 128x128 nearest-neighbor
kappa_cell=kappa_grid(:);                             % cartGrid([128 128]) ordering
fprintf('\n=== SPE10 L%d resampled to 128x128 ===\n',LAYER);
fprintf('kappa range [%.3e %.3e], contrast %.2e x\n',min(kappa_cell),max(kappa_cell),max(kappa_cell)/min(kappa_cell));

G=computeGeometry(cartGrid([128 128],[1 1])); N=G.faces.neighbors;
rock.perm=kappa_cell; rock.poro=ones(G.cells.num,1); G.rock=rock;
hT=computeTrans(G,rock); T_harm=1./accumarray(G.cells.faces(:,1),1./hT,[G.faces.num,1]);
bc=pside(pside([],G,'LEFT',1),G,'RIGHT',4);
state=incompTPFA(initResSol(G,0),G,T_harm,initSingleFluid('mu',1,'rho',1),'bc',bc,'use_trans',true);
divv=accumarray(N(N(:,1)>0,1),state.flux(N(:,1)>0),[G.cells.num,1])-accumarray(N(N(:,2)>0,2),state.flux(N(:,2)>0),[G.cells.num,1]);
fprintf('p range [%.4f %.4f], max|div v|=%.3e (conservative)\n',min(state.pressure),max(state.pressure),max(abs(divv)));

bf0=find(any(N==0,2)); xb0=G.faces.centroids(bf0,1);
Qin=sum(abs(state.flux(bf0(abs(xb0-1)<1e-9)))); pv=poreVolume(G,rock); PV=sum(pv);
T1=0.3*PV/Qin; T2=0.5*PV/Qin;
fprintf('Q_in(x=1)=%.5f -> T(PVI=0.3)=%.5f  T(PVI=0.5)=%.5f\n',Qin,T1,T2);

%% connection list; inflow S=1 at x=1
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

%% MRST explicitTransport (independent)
fluid2=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
bc2=pside([],G,'LEFT',1,'sat',[0 1]); bc2=pside(bc2,G,'RIGHT',4,'sat',[1 0]);
st=incompTPFA(initResSol(G,0,[0 1]),G,T_harm,fluid2,'bc',bc2,'use_trans',true); tE=tic;
st=explicitTransport(st,G,T1,rock,fluid2,'bc',bc2,'Trans',T_harm); s_e_t1=st.s(:,1);
st=explicitTransport(st,G,T2-T1,rock,fluid2,'bc',bc2,'Trans',T_harm); s_e_t2=st.s(:,1);
fprintf('explicitTransport %.1fs\n',toc(tE));
rmse=@(a,b) sqrt(mean((a-b).^2));
fprintf('PVI=0.3: indep max=%.4f hand max=%.4f RMSE=%.3e Sw range [%.4f %.4f]\n',max(s_e_t1),max(s_h_t1),rmse(s_e_t1,s_h_t1),min(s_e_t1),max(s_e_t1));
fprintf('PVI=0.5: indep max=%.4f hand max=%.4f RMSE=%.3e\n',max(s_e_t2),max(s_h_t2),rmse(s_e_t2,s_h_t2));

%% export (NEW file)
p_matrix=state.pressure; xc_matrix=G.cells.centroids;
np=G.faces.nodePos; aF=(1:G.faces.num)'; n1=G.faces.nodes(np(aF)); n2=G.faces.nodes(np(aF)+1);
face_p1=G.nodes.coords(n1,:); face_p2=G.nodes.coords(n2,:); face_centroid=G.faces.centroids;
face_len=G.faces.areas; face_normal=G.faces.normals./face_len; face_flux=state.flux;
face_neighbors=N; face_is_boundary=double(any(N==0,2));
sw_matrix_pvi03=s_e_t1(:); sw_matrix_pvi05=s_e_t2(:);
sw_matrix_pvi03_matched=s_h_t1(:); sw_matrix_pvi05_matched=s_h_t2(:);
meta_source=sprintf('SPE10 model-2 layer %d (Tarbert), Kx normalized geomean=1, resampled 60x220 -> 128x128 nearest',LAYER);
meta_grid='cartGrid([128 128]) unit square; kappa_grid(i,j): i=x, j=y';
meta_bc='p=1 (x=0), p=4 (x=1), no-flow top/bottom; f=0; NO fracture; S=1 inflow at x=1';
meta_layer=LAYER; meta_contrast=max(kappa_cell)/min(kappa_cell); meta_avg='MRST harmonic averaging';
meta_Q_in=Qin; meta_PV=PV; meta_T_pvi03=T1; meta_T_pvi05=T2;
meta_transport_solver='MRST explicitTransport (sw_*) ; hand-coded upwind (sw_*_matched)';
README=sprintf('SPE10 L%d (Tarbert) resampled 128x128. kappa_grid=128x128 normalized perm. face_flux=MRST conservative flux. sw_matrix_pvi03/05=Sw at PVI 0.3/0.5. For CG: 128x128 mesh, load kappa_grid as DG0, BCs p=1/p=4.',LAYER);
save('c:\Users\muchamad\mrst-project\case3_mrst_export_spe10_L20_128.mat','-v7', ...
  'kappa_cell','kappa_grid','p_matrix','xc_matrix','face_p1','face_p2','face_centroid','face_normal', ...
  'face_len','face_flux','face_neighbors','face_is_boundary', ...
  'sw_matrix_pvi03','sw_matrix_pvi05','sw_matrix_pvi03_matched','sw_matrix_pvi05_matched', ...
  'meta_source','meta_grid','meta_bc','meta_layer','meta_contrast','meta_avg','meta_Q_in','meta_PV', ...
  'meta_T_pvi03','meta_T_pvi05','meta_transport_solver','README');
fprintf('saved case3_mrst_export_spe10_L20_128.mat\n');

figure('Name','SPE10 L20 128 log10 kappa','Position',[60 80 500 460]);
plotCellData(G,log10(kappa_cell),'EdgeColor','none'); colormap(jet); view(0,90); axis equal tight; colorbar;
title('SPE10 L20 (128^2)  log_{10}\kappa'); xlabel x; ylabel y;
figure('Name','SPE10 L20 Sw PVI=0.3','Position',[580 80 500 460]);
plotCellData(G,s_e_t1,'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]); view(0,90); axis equal tight; colorbar;
title(sprintf('SPE10 L20  S_w at PVI=0.3 (T=%.4f)',T1)); xlabel x; ylabel y;

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
