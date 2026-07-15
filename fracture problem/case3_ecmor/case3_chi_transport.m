%% chi (top-hat) heterogeneous permeability: 3 interior blocks + 3 inlet strips.
%% Koppel-Martin BCs (p=1 x=0, p=4 x=1, no-flow top/bottom, f=0, no fracture).
%% Harmonic averaging. S=1 inflow at x=1. Two snapshots PVI=0.3, 0.5.
%% Hand-code + MRST explicitTransport. Writes ONLY case3_mrst_export_chi.mat.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
gravity reset off
delta=0.02; CFL=0.45; FPRIME_MAX=2.0;
chi=@(z,a,b,d) 0.5*(tanh((z-a)./d)-tanh((z-b)./d));

G=computeGeometry(cartGrid([128 128],[1 1])); xc=G.cells.centroids; x=xc(:,1); y=xc(:,2); N=G.faces.neighbors;
block1 =chi(x,0.15,0.45,delta).*chi(y,0.55,0.85,delta);
block2 =chi(x,0.55,0.85,delta).*chi(y,0.15,0.40,delta);
channel=chi(x,0.35,0.70,delta).*chi(y,0.40,0.60,delta);
inlet_x=chi(x,0.88,1.00,delta);
inlet_low =inlet_x.*chi(y,0.00,0.25,delta);
inlet_mid =inlet_x.*chi(y,0.38,0.62,delta);
inlet_high=inlet_x.*chi(y,0.78,1.00,delta);
kappa=1 + 9*block1 + 14*block2 + 6*channel + 4*inlet_low + 11*inlet_mid + 19*inlet_high;
fprintf('\n=== chi-perm (delta=%.3f) ===\n',delta);
fprintf('kappa range [%.4f %.4f], contrast %.2f x\n',min(kappa),max(kappa),max(kappa)/min(kappa));

rock.perm=kappa; rock.poro=ones(G.cells.num,1); G.rock=rock;
hT=computeTrans(G,rock); T_harm=1./accumarray(G.cells.faces(:,1),1./hT,[G.faces.num,1]);
bc=pside(pside([],G,'LEFT',1),G,'RIGHT',4);
state=incompTPFA(initResSol(G,0),G,T_harm,initSingleFluid('mu',1,'rho',1),'bc',bc,'use_trans',true);
divv=accumarray(N(N(:,1)>0,1),state.flux(N(:,1)>0),[G.cells.num,1])-accumarray(N(N(:,2)>0,2),state.flux(N(:,2)>0),[G.cells.num,1]);
fprintf('p range [%.4f %.4f], max|div v|=%.3e (conservative)\n',min(state.pressure),max(state.pressure),max(abs(divv)));

bf0=find(any(N==0,2)); xb0=G.faces.centroids(bf0,1);
Qin=sum(abs(state.flux(bf0(abs(xb0-1)<1e-9)))); pv=poreVolume(G,rock); PV=sum(pv);
T1=0.3*PV/Qin; T2=0.5*PV/Qin;
fprintf('Q_in(x=1)=%.5f -> T(PVI=0.3)=%.5f  T(PVI=0.5)=%.5f\n',Qin,T1,T2);

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

fluid2=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
bc2=pside([],G,'LEFT',1,'sat',[0 1]); bc2=pside(bc2,G,'RIGHT',4,'sat',[1 0]);
st=incompTPFA(initResSol(G,0,[0 1]),G,T_harm,fluid2,'bc',bc2,'use_trans',true); tE=tic;
st=explicitTransport(st,G,T1,rock,fluid2,'bc',bc2,'Trans',T_harm); s_e_t1=st.s(:,1);
st=explicitTransport(st,G,T2-T1,rock,fluid2,'bc',bc2,'Trans',T_harm); s_e_t2=st.s(:,1);
fprintf('explicitTransport %.1fs\n',toc(tE)); rmse=@(a,b)sqrt(mean((a-b).^2));
fprintf('PVI=0.3: indep max=%.4f hand max=%.4f RMSE=%.3e Sw range [%.4f %.4f]\n',max(s_e_t1),max(s_h_t1),rmse(s_e_t1,s_h_t1),min(s_e_t1),max(s_e_t1));
fprintf('PVI=0.5: indep max=%.4f hand max=%.4f RMSE=%.3e\n',max(s_e_t2),max(s_h_t2),rmse(s_e_t2,s_h_t2));

%% export (NEW file)
kappa_cell=kappa; kappa_grid=reshape(kappa,[128 128]); p_matrix=state.pressure; xc_matrix=xc;
np=G.faces.nodePos; aF=(1:G.faces.num)'; n1=G.faces.nodes(np(aF)); n2=G.faces.nodes(np(aF)+1);
face_p1=G.nodes.coords(n1,:); face_p2=G.nodes.coords(n2,:); face_centroid=G.faces.centroids;
face_len=G.faces.areas; face_normal=G.faces.normals./face_len; face_flux=state.flux;
face_neighbors=N; face_is_boundary=double(any(N==0,2));
sw_matrix_pvi03=s_e_t1(:); sw_matrix_pvi05=s_e_t2(:);
sw_matrix_pvi03_matched=s_h_t1(:); sw_matrix_pvi05_matched=s_h_t2(:);
meta_kappa='1+9*chi[.15,.45]x*chi[.55,.85]y +14*chi[.55,.85]x*chi[.15,.40]y +6*chi[.35,.70]x*chi[.40,.60]y +4*chi[.88,1]x*chi[0,.25]y +11*chi[.88,1]x*chi[.38,.62]y +19*chi[.88,1]x*chi[.78,1]y; chi=0.5(tanh((z-a)/d)-tanh((z-b)/d)), d=0.02';
meta_bc='p=1 (x=0), p=4 (x=1), no-flow top/bottom; f=0; NO fracture; S=1 inflow at x=1';
meta_delta=delta; meta_contrast=max(kappa)/min(kappa); meta_avg='MRST harmonic averaging';
meta_Q_in=Qin; meta_PV=PV; meta_T_pvi03=T1; meta_T_pvi05=T2;
meta_transport_solver='MRST explicitTransport (sw_*) ; hand-coded upwind (sw_*_matched)';
README='chi top-hat heterogeneous case (3 blocks + 3 inlet strips, delta=0.02, max kappa~20). kappa_grid=128x128 perm. face_flux=MRST conservative flux. sw_matrix_pvi03/05=Sw at PVI 0.3/0.5. For CG: 128x128 mesh, load kappa_grid as DG0, BCs p=1/p=4.';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_chi.mat','-v7', ...
  'kappa_cell','kappa_grid','p_matrix','xc_matrix','face_p1','face_p2','face_centroid','face_normal', ...
  'face_len','face_flux','face_neighbors','face_is_boundary', ...
  'sw_matrix_pvi03','sw_matrix_pvi05','sw_matrix_pvi03_matched','sw_matrix_pvi05_matched', ...
  'meta_kappa','meta_bc','meta_delta','meta_contrast','meta_avg','meta_Q_in','meta_PV', ...
  'meta_T_pvi03','meta_T_pvi05','meta_transport_solver','README');
fprintf('saved case3_mrst_export_chi.mat\n');

figure('Name','chi kappa','Position',[60 90 480 440]);
plotCellData(G,kappa,'EdgeColor','none'); colormap(turbo); view(0,90); axis equal tight; colorbar;
title(sprintf('\\kappa (chi, \\delta=%.2f, max=%.1f)',delta,max(kappa))); xlabel x; ylabel y;
figure('Name','chi Sw PVI=0.3','Position',[560 90 480 440]);
plotCellData(G,s_e_t1,'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]); view(0,90); axis equal tight; colorbar;
title(sprintf('S_w at PVI=0.3 (T=%.4f)',T1)); xlabel x; ylabel y;

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
