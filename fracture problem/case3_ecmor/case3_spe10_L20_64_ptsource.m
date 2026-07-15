%% SPE10 L20 64x64, KM pressure (unchanged, f=0) but TRANSPORT source = interior point.
%% S=1 held at the cell nearest (0.5,0.8); x=1 boundary now brings oil (S=0).
%% Flow field identical to case3_spe10_L20_64.m; only the saturation source changes.
%% Hand-coded upwind only (explicitTransport can't do interior Dirichlet-S with f=0).
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add spe10 incomp
gravity reset off
LAYER=20; NG=64; CFL=0.45; FPRIME_MAX=2.0;
XSRC=0.9; YSRC=0.5; Tsnaps=[0.20 0.40];   % point-source location + snapshot times (up to T=0.4)

%% permeability (same as case3_spe10_L20_64)
rk=getSPE10rock(LAYER); Kx=rk.perm(:,1); kappa=Kx./exp(mean(log(Kx)));
Korig=reshape(kappa,[60 220]);
ispe=min(60,max(1,ceil(((1:NG)-0.5)/NG*60)));
jspe=min(220,max(1,ceil(((1:NG)-0.5)/NG*220)));
kappa_grid=Korig(ispe,jspe); kappa_cell=kappa_grid(:);
fprintf('\n=== SPE10 L%d %dx%d, POINT-SOURCE transport at (%.2f,%.2f) ===\n',LAYER,NG,NG,XSRC,YSRC);

G=computeGeometry(cartGrid([NG NG],[1 1])); N=G.faces.neighbors;
rock.perm=kappa_cell; rock.poro=ones(G.cells.num,1); G.rock=rock;
hT=computeTrans(G,rock); T_harm=1./accumarray(G.cells.faces(:,1),1./hT,[G.faces.num,1]);
bc=pside(pside([],G,'LEFT',1),G,'RIGHT',4);
state=incompTPFA(initResSol(G,0),G,T_harm,initSingleFluid('mu',1,'rho',1),'bc',bc,'use_trans',true);
divv=accumarray(N(N(:,1)>0,1),state.flux(N(:,1)>0),[G.cells.num,1])-accumarray(N(N(:,2)>0,2),state.flux(N(:,2)>0),[G.cells.num,1]);
fprintf('p range [%.4f %.4f], max|div v|=%.3e (f=0, flow unchanged)\n',min(state.pressure),max(state.pressure),max(abs(divv)));

%% distances to the requested point (source picked after fluxes are known)
d2=(G.cells.centroids(:,1)-XSRC).^2+(G.cells.centroids(:,2)-YSRC).^2;

%% connection list (boundary inflow = oil S=0; interior point held S=1)
isInt=N(:,1)>0&N(:,2)>0; oI=N(isInt,1); nI=N(isInt,2); FI=state.flux(isInt);
bf=find(~isInt); ownB=max(N(bf,1),N(bf,2)); sgnB=ones(numel(bf),1); sgnB(N(bf,1)==0)=-1;
FB=state.flux(bf).*sgnB;
owner=[oI;ownB]; neigh=[nI;-ones(numel(bf),1)]; Fout=[FI;FB]; hasNb=neigh>0;
pv=poreVolume(G,rock);
outflux=accumarray(owner,max(Fout,0),[G.cells.num,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[G.cells.num,1]);
act=outflux>1e-30; dt_cfl=CFL*min(pv(act)./(FPRIME_MAX*outflux(act)));

%% source = strongest-FLOW cell within RAD of the requested point (low-kappa cells pass no plume)
RAD=0.08; cand=find(d2<RAD^2); [~,im]=max(outflux(cand)); srcCell=cand(im);
sc=G.cells.centroids(srcCell,:);
fprintf('source cell %d at (%.4f,%.4f) [max-flow within %.2f of (%.2f,%.2f)], kappa=%.4g, outflux=%.4g\n',srcCell,sc(1),sc(2),RAD,XSRC,YSRC,kappa_cell(srcCell),outflux(srcCell));

S1=march_pt(Tsnaps(1),dt_cfl,Fout,owner,neigh,hasNb,pv,G.cells.num,srcCell);
S2=march_pt(Tsnaps(2),dt_cfl,Fout,owner,neigh,hasNb,pv,G.cells.num,srcCell);

%% export (NEW file)
p_matrix=state.pressure; xc_matrix=G.cells.centroids;
np=G.faces.nodePos; aF=(1:G.faces.num)'; n1=G.faces.nodes(np(aF)); n2=G.faces.nodes(np(aF)+1);
face_p1=G.nodes.coords(n1,:); face_p2=G.nodes.coords(n2,:); face_centroid=G.faces.centroids;
face_len=G.faces.areas; face_normal=G.faces.normals./face_len; face_flux=state.flux;
face_neighbors=N; face_is_boundary=double(any(N==0,2));
sw_ptsrc_T1=S1(:); sw_ptsrc_T2=S2(:);
src_cell=srcCell; src_xy=sc; src_kappa=kappa_cell(srcCell); snap_times=Tsnaps;
meta_source=sprintf('SPE10 L%d (Tarbert) normalized, resampled 60x220 -> %dx%d nearest',LAYER,NG,NG);
meta_bc=sprintf('KM pressure p=1(x=0)/p=4(x=1), no-flow top/bottom, f=0. Transport: x=1 boundary S=0 (oil); interior point (%.2f,%.2f) held S=1.',XSRC,YSRC);
meta_note='Point-source is a Dirichlet-S=1 held cell (transport-side source, no volume source, flow unchanged). Hand-code upwind only.';
meta_T1=Tsnaps(1); meta_T2=Tsnaps(2); meta_avg='MRST harmonic averaging';
README=sprintf('SPE10 L20 64x64, interior point-source transport. Flow = KM pressure (f=0). sw_ptsrc_T1/T2 = Sw at snap_times. src_cell/src_xy mark the S=1 point. face_flux=MRST conservative flux. CG: same 64x64 mesh + kappa_grid + KM BCs, hold S=1 at the cell nearest (%.2f,%.2f), boundary inflow S=0.',XSRC,YSRC);
save('c:\Users\muchamad\mrst-project\case3_mrst_export_spe10_L20_64_ptsource.mat','-v7', ...
  'kappa_cell','kappa_grid','p_matrix','xc_matrix','face_p1','face_p2','face_centroid','face_normal', ...
  'face_len','face_flux','face_neighbors','face_is_boundary','sw_ptsrc_T1','sw_ptsrc_T2', ...
  'src_cell','src_xy','src_kappa','snap_times','meta_source','meta_bc','meta_note','meta_T1','meta_T2','meta_avg','README');
fprintf('saved case3_mrst_export_spe10_L20_64_ptsource.mat\n');

%% figures: perm map + saturation snapshots (source marked)
figure('Name','SPE10 64 ptsrc log10 kappa','Position',[40 90 460 430]);
plotCellData(G,log10(kappa_cell),'EdgeColor','none'); colormap(jet); view(0,90); axis equal tight; colorbar;
hold on; plot(sc(1),sc(2),'wp','MarkerSize',14,'MarkerFaceColor','w');
title('SPE10 L20 (64^2)  log_{10}\kappa  + source \bigstar'); xlabel x; ylabel y;
figure('Name','SPE10 64 ptsrc Sw t1','Position',[510 90 460 430]);
plotCellData(G,S1,'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]); view(0,90); axis equal tight; colorbar;
hold on; plot(sc(1),sc(2),'rp','MarkerSize',14,'MarkerFaceColor','r');
title(sprintf('S_w at T=%.2f (point source)',Tsnaps(1))); xlabel x; ylabel y;
figure('Name','SPE10 64 ptsrc Sw t2','Position',[980 90 460 430]);
plotCellData(G,S2,'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]); view(0,90); axis equal tight; colorbar;
hold on; plot(sc(1),sc(2),'rp','MarkerSize',14,'MarkerFaceColor','r');
title(sprintf('S_w at T=%.2f (point source)',Tsnaps(2))); xlabel x; ylabel y;

function S=march_pt(Ttar,dt_cfl,Fout,owner,neigh,hasNb,pv,ncell,srcCell)
  fbl=@(s)s.^2./(s.^2+(1-s).^2+1e-30);
  nsteps=max(1,ceil(Ttar/dt_cfl)); dt=Ttar/nsteps; nConn=numel(Fout);
  outfl=Fout>=0; isBin=~outfl&~hasNb; upCell=ones(nConn,1);
  upCell(outfl)=owner(outfl); sel=~outfl&hasNb; upCell(sel)=neigh(sel);
  hb=find(hasNb);
  Inc=sparse([owner;neigh(hb)],[(1:nConn)';hb],[ones(nConn,1);-ones(numel(hb),1)],ncell,nConn);
  dt_pv=dt./pv; S=zeros(ncell,1); S(srcCell)=1;
  for it=1:nsteps
    w=fbl(S(upCell)).*Fout; w(isBin)=0;   % boundary inflow = oil (S=0)
    S=min(max(S-dt_pv.*(Inc*w),0),1);
    S(srcCell)=1;                          % hold interior point at S=1
  end
  fprintf('  march to T=%.4f: %d steps, Sw_max=%.4f, plume cells (S>0.5)=%d\n',Ttar,nsteps,max(S),nnz(S>0.5));
end
