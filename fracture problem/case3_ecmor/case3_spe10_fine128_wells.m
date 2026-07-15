%% FINE 128x128 uniform grid (h/2=1/128) resolving BOTH 64x64 primal and 65x65 dual exactly.
%% kappa_fine = kron(kappa_primal, ones(2,2)) -> each fine cell = one primal SPE10 cell (NO averaging).
%% Sources are COARSE 1/64 cells -> each maps to a 2x2 block of 4 fine cells:
%%   injector +1 -> +0.25 on each of 4 fine cells (S=1); producer -5 -> -1.25 on each of 4 (q_w=rate*F(S)).
%% -div(kappa grad p)=q, p=0 on all dOmega. Transport: boundary inflow S=0, S0=0. Snapshots 0.02/0.05/0.10.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add spe10 incomp
gravity reset off
LAYER=20; NG=64; NF=128; CFL=0.45; FPRIME_MAX=2.0;
qI=1; qP=-5; Tsnaps=[0.02 0.05 0.10];
inj_xy=[0.2 0.4]; prod_xy=[0.8 0.5];
inj_reg=[0.1953125 0.3984375]; prod_reg=[0.8046875 0.5078125];   % coarse (1/64) cell centers

%% primal 64x64 SPE10 -> exact 2x2 replication onto 128x128 fine cells
rk=getSPE10rock(LAYER); Kx=rk.perm(:,1); kappa=Kx./exp(mean(log(Kx)));
Korig=reshape(kappa,[60 220]);
ispe=min(60,max(1,ceil(((1:NG)-0.5)/NG*60)));
jspe=min(220,max(1,ceil(((1:NG)-0.5)/NG*220)));
Kp=Korig(ispe,jspe);
kappa_fine_grid=kron(Kp,ones(2,2)); kappa_cell=kappa_fine_grid(:);
fprintf('\n=== FINE 128x128 (exact kappa), 4-fine-cell wells, p=0 ===\n');
fprintf('fine kappa range [%.3e %.3e] == primal (contrast %.2e x, not compressed)\n', ...
        min(kappa_cell),max(kappa_cell),max(kappa_cell)/min(kappa_cell));

G=computeGeometry(cartGrid([NF NF],[1 1])); N=G.faces.neighbors; nc=G.cells.num;
cx=G.cells.centroids(:,1); cy=G.cells.centroids(:,2);
rock.perm=kappa_cell; rock.poro=ones(nc,1); G.rock=rock;
hT=computeTrans(G,rock); T_harm=1./accumarray(G.cells.faces(:,1),1./hT,[G.faces.num,1]);

%% coarse 1/64 source cell -> its 2x2 block of fine cells
pmI=round(inj_reg*NG+0.5); pmP=round(prod_reg*NG+0.5);        % [13 26], [52 33]
fine4=@(pm) reshape(((2*pm(1)-1:2*pm(1))') + ((2*pm(2)-1:2*pm(2))-1)*NF, [],1);
injCells=fine4(pmI); prodCells=fine4(pmP);
fprintf('injector primal (%d,%d) -> fine cells [%s]\n',pmI(1),pmI(2),num2str(injCells'));
fprintf('producer primal (%d,%d) -> fine cells [%s]\n',pmP(1),pmP(2),num2str(prodCells'));

%% pressure: p=0 all boundary faces + distributed sources
bf=boundaryFaces(G); bc=addBC([],bf,'pressure',zeros(numel(bf),1),'sat',[0 1]);
src=addSource([],injCells,repmat(qI/numel(injCells),numel(injCells),1),'sat',repmat([1 0],numel(injCells),1));
src=addSource(src,prodCells,repmat(qP/numel(prodCells),numel(prodCells),1),'sat',repmat([0 1],numel(prodCells),1));
fluid=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
state=incompTPFA(initResSol(G,0,[0 1]),G,T_harm,fluid,'bc',bc,'src',src,'use_trans',true);
divv=accumarray(N(N(:,1)>0,1),state.flux(N(:,1)>0),[nc,1])-accumarray(N(N(:,2)>0,2),state.flux(N(:,2)>0),[nc,1]);
resid=divv; resid([injCells;prodCells])=0;
fprintf('p range [%.4f %.4f]; sum div v inj=%.4f prod=%.4f; max|div v| elsewhere=%.3e\n', ...
        min(state.pressure),max(state.pressure),sum(divv(injCells)),sum(divv(prodCells)),max(abs(resid)));

%% connection list + CFL
isInt=N(:,1)>0&N(:,2)>0; oI=N(isInt,1); nI=N(isInt,2); FI=state.flux(isInt);
bfc=find(~isInt); ownB=max(N(bfc,1),N(bfc,2)); sgnB=ones(numel(bfc),1); sgnB(N(bfc,1)==0)=-1;
FB=state.flux(bfc).*sgnB;
owner=[oI;ownB]; neigh=[nI;-ones(numel(bfc),1)]; Fout=[FI;FB]; hasNb=neigh>0;
pv=poreVolume(G,rock);
outflux=accumarray(owner,max(Fout,0),[nc,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[nc,1]);
outflux(injCells)=outflux(injCells)+abs(qI/numel(injCells)); outflux(prodCells)=outflux(prodCells)+abs(qP/numel(prodCells));
act=outflux>1e-30; dt_cfl=CFL*min(pv(act)./(FPRIME_MAX*outflux(act)));
fprintf('PV=%.5f, dt_cfl=%.3e\n',sum(pv),dt_cfl);

[S1,k1,d1]=march_wells(zeros(nc,1),Tsnaps(1),          dt_cfl,Fout,owner,neigh,hasNb,pv,nc,injCells,prodCells,qI,qP);
[S2,k2,d2]=march_wells(S1,        Tsnaps(2)-Tsnaps(1), dt_cfl,Fout,owner,neigh,hasNb,pv,nc,injCells,prodCells,qI,qP);
[S3,k3,d3]=march_wells(S2,        Tsnaps(3)-Tsnaps(2), dt_cfl,Fout,owner,neigh,hasNb,pv,nc,injCells,prodCells,qI,qP);
cum_nsteps=[k1 k1+k2 k1+k2+k3];

haveExp=false; se1=[]; se2=[]; se3=[];
try
  st=state; tE=tic;
  st=explicitTransport(st,G,Tsnaps(1),rock,fluid,'bc',bc,'src',src,'Trans',T_harm); se1=st.s(:,1);
  st=explicitTransport(st,G,Tsnaps(2)-Tsnaps(1),rock,fluid,'bc',bc,'src',src,'Trans',T_harm); se2=st.s(:,1);
  st=explicitTransport(st,G,Tsnaps(3)-Tsnaps(2),rock,fluid,'bc',bc,'src',src,'Trans',T_harm); se3=st.s(:,1);
  haveExp=true;
  fprintf('explicitTransport OK (%.1fs): T=0.05 indep max=%.4f RMSE vs hand=%.3e\n',toc(tE),max(se2),sqrt(mean((se2-S2).^2)));
catch ME
  fprintf('explicitTransport unavailable (%s) -> hand-code only\n',ME.message);
end

%% export
xc_matrix=G.cells.centroids; p_matrix=state.pressure;
np=G.faces.nodePos; aF=(1:G.faces.num)'; nn1=G.faces.nodes(np(aF)); nn2=G.faces.nodes(np(aF)+1);
face_p1=G.nodes.coords(nn1,:); face_p2=G.nodes.coords(nn2,:); face_centroid=G.faces.centroids;
face_len=G.faces.areas; face_normal=G.faces.normals./face_len; face_flux=state.flux;
face_neighbors=N; face_is_boundary=double(any(N==0,2));
sw_T002=S1(:); sw_T005=S2(:); sw_T010=S3(:); snap_times=Tsnaps;
inj_cell=injCells; prod_cell=prodCells; inj_primal=pmI; prod_primal=pmP; q_inj=qI; q_prod=qP;
meta_dt=[d1 d2 d3]; meta_nsteps=cum_nsteps; meta_CFL=CFL; meta_FPRIME_MAX=FPRIME_MAX;
if haveExp, sw_T002_exp=se1(:); sw_T005_exp=se2(:); sw_T010_exp=se3(:); else, sw_T002_exp=[]; sw_T005_exp=[]; sw_T010_exp=[]; end
meta_grid='FINE 128x128 uniform; resolves 64x64 primal and 65x65 dual exactly.';
meta_kappa='kappa_fine=kron(kappa_primal_64,ones(2,2)) -> each fine cell = one primal SPE10 cell (EXACT, no averaging).';
meta_bc='p=0 on ALL boundaries. Sources = COARSE 1/64 cells, each split over its 2x2 block of 4 fine cells: injector +0.25/cell S=1, producer -1.25/cell q_w=rate*F(S). Boundary inflow S=0.';
meta_note='inj_cell/prod_cell are the 4 fine cells per well. To compare on Deng dual mesh: sum fine face-fluxes over each dual face.';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_fine128_wells.mat','-v7', ...
  'xc_matrix','p_matrix','face_flux','face_neighbors','face_centroid','face_normal','face_len', ...
  'face_p1','face_p2','face_is_boundary','kappa_cell','sw_T002','sw_T005','sw_T010','snap_times', ...
  'inj_cell','prod_cell','inj_primal','prod_primal','inj_xy','prod_xy','q_inj','q_prod', ...
  'meta_dt','meta_nsteps','meta_CFL','meta_FPRIME_MAX','sw_T002_exp','sw_T005_exp','sw_T010_exp', ...
  'kappa_fine_grid','meta_grid','meta_kappa','meta_bc','meta_note');
fprintf('saved case3_mrst_export_fine128_wells.mat\n');

figure('Name','fine128 log10 kappa','Position',[30 100 460 430]);
plotCellData(G,log10(kappa_cell),'EdgeColor','none'); colormap(jet); view(0,90); axis equal tight; colorbar; hold on;
markwells(inj_reg(1),inj_reg(2),prod_reg(1),prod_reg(2));
title('fine 128^2 log_{10}\kappa (exact, no avg)'); xlabel x; ylabel y;
figure('Name','fine128 pressure','Position',[500 100 460 430]);
plotCellData(G,state.pressure,'EdgeColor','none'); colormap(parula); view(0,90); axis equal tight; colorbar; hold on;
markwells(inj_reg(1),inj_reg(2),prod_reg(1),prod_reg(2));
title('pressure p (p=0 on \partial\Omega)'); xlabel x; ylabel y;
figure('Name','fine128 Sw snapshots','Position',[100 60 1180 400]);
SS={S1,S2,S3};
for k=1:3
  subplot(1,3,k);
  plotCellData(G,SS{k},'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]); view(0,90); axis equal tight; hold on;
  markwells(inj_reg(1),inj_reg(2),prod_reg(1),prod_reg(2));
  lab=''; if k==2, lab=' (selected)'; end
  title(sprintf('S_w at T=%.2f%s',Tsnaps(k),lab)); xlabel x; ylabel y;
end
colorbar('Position',[0.93 0.15 0.015 0.7]);

function markwells(xi,yi,xp,yp)
  plot(xi,yi,'g^','MarkerSize',12,'MarkerFaceColor','g','LineWidth',1);
  plot(xp,yp,'rv','MarkerSize',12,'MarkerFaceColor','r','LineWidth',1);
end

function [S,nsteps,dt]=march_wells(S,Tdur,dt_cfl,Fout,owner,neigh,hasNb,pv,ncell,injCells,prodCells,qI,qP)
  fbl=@(s)s.^2./(s.^2+(1-s).^2+1e-30);
  nsteps=max(1,ceil(Tdur/dt_cfl)); dt=Tdur/nsteps; nConn=numel(Fout);
  outfl=Fout>=0; isBin=~outfl&~hasNb; upCell=ones(nConn,1);
  upCell(outfl)=owner(outfl); sel=~outfl&hasNb; upCell(sel)=neigh(sel);
  hb=find(hasNb);
  Inc=sparse([owner;neigh(hb)],[(1:nConn)';hb],[ones(nConn,1);-ones(numel(hb),1)],ncell,nConn);
  dt_pv=dt./pv; qiPer=qI/numel(injCells); qpPer=qP/numel(prodCells);
  for it=1:nsteps
    w=fbl(S(upCell)).*Fout; w(isBin)=0;
    A=Inc*w; qw=zeros(ncell,1); qw(injCells)=qiPer; qw(prodCells)=qpPer*fbl(S(prodCells));
    S=min(max(S+dt_pv.*(-A+qw),0),1);
  end
  fprintf('  +%.4f (%d steps, dt=%.3e): Sw_max=%.4f, cells S>0.5=%d\n',Tdur,nsteps,dt,max(S),nnz(S>0.5));
end
