%% FINE pressure (exact kappa, 128x128) -> aggregate flux to DUAL faces -> DUAL-mesh transport.
%% Source convention MATCHES the notebook:
%%   PRESSURE source = ONE primal 64x64 cell (K_I,K_P), realized as its 2x2 fine cells (+0.25/-1.25 each).
%%   TRANSPORT source = split over the 4 corner-node DUAL CVs of that primal cell (+0.25/-1.25 each).
%%   The primal cell's 4 fine cells map 1:1 onto those 4 dual CVs -> div v_dual = q_dual cell-by-cell.
%% Injector S=1; producer removes at local F(S); boundary inflow S=0; F=S^2/(S^2+(1-S)^2). phi=1.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add spe10 incomp
gravity reset off
LAYER=20; NG=64; NF=128; CFL=0.45; FPRIME_MAX=2.0;
qI=1; qP=-5; Tsnaps=[0.02 0.05 0.10];
inj_reg=[0.1953125 0.3984375]; prod_reg=[0.8046875 0.5078125];   % primal 64x64 cell centers (1-based (13,26),(52,33))
h=1/NG; dx=[h/2, h*ones(1,NG-1), h/2]; dy=dx; xv=[0 cumsum(dx)]; yv=[0 cumsum(dy)];

%% fine grid + EXACT kappa
Gf=computeGeometry(cartGrid([NF NF],[1 1])); Nf=Gf.faces.neighbors; ncf=Gf.cells.num;
rk=getSPE10rock(LAYER); Kx=rk.perm(:,1); kappa=Kx./exp(mean(log(Kx)));
Korig=reshape(kappa,[60 220]);
ispe=min(60,max(1,ceil(((1:NG)-0.5)/NG*60))); jspe=min(220,max(1,ceil(((1:NG)-0.5)/NG*220)));
Kp=Korig(ispe,jspe); kappa_fine_cell=reshape(kron(Kp,ones(2,2)),[],1);
rockf.perm=kappa_fine_cell; rockf.poro=ones(ncf,1);
hTf=computeTrans(Gf,rockf); Tf=1./accumarray(Gf.cells.faces(:,1),1./hTf,[Gf.faces.num,1]);

%% dual grid (65x65, half-size boundary cells)
Gd=computeGeometry(tensorGrid(xv,yv)); Nd=Gd.faces.neighbors; ncd=Gd.cells.num;

%% PRIMAL source cells -> their 2x2 fine cells (for pressure)
pmI=round(inj_reg*NG+0.5); pmP=round(prod_reg*NG+0.5);          % 1-based primal cells
fsP=@(m)[2*m-1, 2*m];                                           % primal cell -> its 2 fine cells (1D)
[a,b]=ndgrid(fsP(pmI(1)),fsP(pmI(2))); injFine=a(:)+(b(:)-1)*NF;
[a,b]=ndgrid(fsP(pmP(1)),fsP(pmP(2))); prodFine=a(:)+(b(:)-1)*NF;
fprintf('\n=== fine pressure -> dual transport (notebook-matched source) ===\n');
fprintf('injector primal (%d,%d) center (%.4f,%.4f); fine cells [%s]\n',pmI(1),pmI(2),inj_reg(1),inj_reg(2),num2str(injFine'));
fprintf('producer primal (%d,%d) center (%.4f,%.4f); fine cells [%s]\n',pmP(1),pmP(2),prod_reg(1),prod_reg(2),num2str(prodFine'));

%% fine pressure solve: source on the PRIMAL cells' fine cells, p=0 all boundaries
bff=boundaryFaces(Gf); bcf=addBC([],bff,'pressure',zeros(numel(bff),1),'sat',[0 1]);
srcf=addSource([],injFine,repmat(qI/4,4,1),'sat',repmat([1 0],4,1));
srcf=addSource(srcf,prodFine,repmat(qP/4,4,1),'sat',repmat([0 1],4,1));
fluid=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
statef=incompTPFA(initResSol(Gf,0,[0 1]),Gf,Tf,fluid,'bc',bcf,'src',srcf,'use_trans',true);
fprintf('fine p range [%.4f %.4f]\n',min(statef.pressure),max(statef.pressure));

%% partition fine->dual, then AGGREGATE flux (SUM fine sub-faces per dual face)
[KX,KY]=ndgrid(1:NF,1:NF); pmap=reshape(ceil((KX+1)/2)+(ceil((KY+1)/2)-1)*(NG+1),[],1);
injDualCVs=unique(pmap(injFine)); prodDualCVs=unique(pmap(prodFine));   % the 4 corner-node dual CVs
fprintf('injector dual CVs [%s] centers:\n',num2str(injDualCVs')); disp(Gd.cells.centroids(injDualCVs,:));
fprintf('producer dual CVs [%s]\n',num2str(prodDualCVs'));
isintf=all(Nf>0,2); di=pmap(Nf(isintf,:)); cross=di(:,1)~=di(:,2);
fint=find(isintf); fcr=fint(cross); dc=di(cross,:);
isintd=all(Nd>0,2); dfid=find(isintd);
L=sparse(min(Nd(isintd,:),[],2),max(Nd(isintd,:),[],2),dfid,ncd,ncd);
dfi=full(L(sub2ind([ncd ncd],min(dc,[],2),max(dc,[],2))));
sgn=2*(dc(:,1)==Nd(dfi,1))-1;
Qd=accumarray(dfi,statef.flux(fcr).*sgn,[Gd.faces.num,1]);
% boundary faces -> dual boundary faces (match by dual cell + side)
sideOf=@(c)((abs(c(:,1))<1e-9)+2*(abs(c(:,1)-1)<1e-9)+3*(abs(c(:,2))<1e-9)+4*(abs(c(:,2)-1)<1e-9));
fbf=find(~isintf); own=max(Nf(fbf,:),[],2); dcown=pmap(own);
outsgn=ones(numel(fbf),1); outsgn(Nf(fbf,1)==0)=-1; foutw=statef.flux(fbf).*outsgn;
dbf=find(~isintd); dbcown=max(Nd(~isintd,:),[],2);
Ld=sparse(dbcown,sideOf(Gd.faces.centroids(dbf,:)),dbf,ncd,4);
dbi=full(Ld(sub2ind([ncd 4],dcown,sideOf(Gf.faces.centroids(fbf,:)))));
Foutd=accumarray(dbi,foutw,[Gd.faces.num,1]);
Qb=zeros(Gd.faces.num,1); Qb(Nd(:,2)==0)=Foutd(Nd(:,2)==0); Qb(Nd(:,1)==0)=-Foutd(Nd(:,1)==0);
Qd=Qd+Qb;
% conservation check: div v_dual should be +0.25 on each injector CV, -1.25 on each producer CV
divd=accumarray(Nd(Nd(:,1)>0,1),Qd(Nd(:,1)>0),[ncd,1])-accumarray(Nd(Nd(:,2)>0,2),Qd(Nd(:,2)>0),[ncd,1]);
q_dual=zeros(ncd,1); q_dual(injDualCVs)=qI/4; q_dual(prodDualCVs)=qP/4;
cons_resid=max(abs(divd-q_dual));
fprintf('AGGREGATION CHECK: max|div v_dual - q_dual| = %.3e ; div on inj CVs = [%s]\n',cons_resid,num2str(divd(injDualCVs)',' %.3f'));

%% transport on the DUAL mesh with the aggregated flux Qd; source split over 4 dual CVs per well
isIntd=all(Nd>0,2); oI=Nd(isIntd,1); nI=Nd(isIntd,2); FI=Qd(isIntd);
bfd=find(~isIntd); ownB=max(Nd(bfd,1),Nd(bfd,2)); sgnB=ones(numel(bfd),1); sgnB(Nd(bfd,1)==0)=-1; FB=Qd(bfd).*sgnB;
owner=[oI;ownB]; neigh=[nI;-ones(numel(bfd),1)]; Fout=[FI;FB]; hasNb=neigh>0;
pvd=Gd.cells.volumes;
outflux=accumarray(owner,max(Fout,0),[ncd,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[ncd,1]);
outflux(injDualCVs)=outflux(injDualCVs)+abs(qI/4); outflux(prodDualCVs)=outflux(prodDualCVs)+abs(qP/4);
act=outflux>1e-30; dt_cfl=CFL*min(pvd(act)./(FPRIME_MAX*outflux(act)));
fprintf('dual PV=%.5f, dt_cfl=%.3e\n',sum(pvd),dt_cfl);
[S1,k1,d1]=march_wells(zeros(ncd,1),Tsnaps(1),          dt_cfl,Fout,owner,neigh,hasNb,pvd,ncd,injDualCVs,prodDualCVs,qI,qP);
[S2,k2,d2]=march_wells(S1,        Tsnaps(2)-Tsnaps(1), dt_cfl,Fout,owner,neigh,hasNb,pvd,ncd,injDualCVs,prodDualCVs,qI,qP);
[S3,k3,d3]=march_wells(S2,        Tsnaps(3)-Tsnaps(2), dt_cfl,Fout,owner,neigh,hasNb,pvd,ncd,injDualCVs,prodDualCVs,qI,qP);
cum_nsteps=[k1 k1+k2 k1+k2+k3];

%% export (overwrite case3_mrst_export_finep_dualT.mat)
xc_matrix=Gd.cells.centroids;
np=Gd.faces.nodePos; aF=(1:Gd.faces.num)'; nn1=Gd.faces.nodes(np(aF)); nn2=Gd.faces.nodes(np(aF)+1);
face_p1=Gd.nodes.coords(nn1,:); face_p2=Gd.nodes.coords(nn2,:); face_centroid=Gd.faces.centroids;
face_len=Gd.faces.areas; face_normal=Gd.faces.normals./face_len; face_flux=Qd;
face_neighbors=Nd; face_is_boundary=double(any(Nd==0,2));
sw_T002=S1(:); sw_T005=S2(:); sw_T010=S3(:); snap_times=Tsnaps;
inj_cell=injDualCVs; prod_cell=prodDualCVs; inj_primal=pmI; prod_primal=pmP; q_inj=qI; q_prod=qP;
inj_fine=injFine; prod_fine=prodFine; inj_xy=[0.2 0.4]; prod_xy=[0.8 0.5];
p_matrix_fine=statef.pressure; face_flux_fine=statef.flux;
meta_dt=[d1 d2 d3]; meta_nsteps=cum_nsteps; meta_CFL=CFL; meta_FPRIME_MAX=FPRIME_MAX; conservation_residual=cons_resid;
meta_workflow='128x128 fine pressure (EXACT kappa), source on ONE primal cell (its 4 fine cells, +0.25/-1.25); flux SUMMED over dual faces -> conservative dual flux; transport on 65x65 dual mesh with source split over the 4 corner-node dual CVs (+0.25/-1.25). div v_dual=q_dual cell-by-cell.';
meta_grid='transport/export mesh = 65x65 Deng dual (tensorGrid, half-size boundary cells). face_flux is on THIS dual mesh.';
meta_source='pressure: primal cells (1-based) inj (13,26) prod (52,33), centers (0.1953,0.3984)/(0.8047,0.5078). transport: 4 corner-node dual CVs each, +0.25 (inj S=1) / -1.25 (prod, local F(S)). boundary inflow S=0.';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_finep_dualT.mat','-v7', ...
  'xc_matrix','p_matrix_fine','face_flux','face_neighbors','face_centroid','face_normal','face_len', ...
  'face_p1','face_p2','face_is_boundary','kappa_fine_cell','face_flux_fine','sw_T002','sw_T005','sw_T010','snap_times', ...
  'inj_cell','prod_cell','inj_primal','prod_primal','inj_fine','prod_fine','inj_xy','prod_xy','q_inj','q_prod', ...
  'meta_dt','meta_nsteps','meta_CFL','meta_FPRIME_MAX','conservation_residual','meta_workflow','meta_grid','meta_source');
fprintf('saved case3_mrst_export_finep_dualT.mat\n');

%% figures
figure('Name','finep-dualT fine log10 kappa','Position',[30 100 460 430]);
plotCellData(Gf,log10(kappa_fine_cell),'EdgeColor','none'); colormap(jet); view(0,90); axis equal tight; colorbar;
title('fine 128^2 log_{10}\kappa'); xlabel x; ylabel y;
figure('Name','finep-dualT fine pressure','Position',[500 100 460 430]);
plotCellData(Gf,statef.pressure,'EdgeColor','none'); colormap(parula); view(0,90); axis equal tight; colorbar;
title('fine pressure p'); xlabel x; ylabel y;
figure('Name','finep-dualT dual Sw','Position',[100 60 1180 400]);
SS={S1,S2,S3};
for k=1:3
  subplot(1,3,k);
  plotCellData(Gd,SS{k},'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]); view(0,90); axis equal tight; hold on;
  plot(inj_xy(1),inj_xy(2),'g^','MarkerSize',11,'MarkerFaceColor','g'); plot(prod_xy(1),prod_xy(2),'rv','MarkerSize',11,'MarkerFaceColor','r');
  lab=''; if k==2, lab=' (selected)'; end
  title(sprintf('dual S_w at T=%.2f%s',Tsnaps(k),lab)); xlabel x; ylabel y;
end
colorbar('Position',[0.93 0.15 0.015 0.7]);

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
