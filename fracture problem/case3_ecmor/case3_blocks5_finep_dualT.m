%% 5x5 DETERMINISTIC block permeability + Deng homogeneous-Dirichlet injector/producer setup.
%% Same corrected pipeline as case3_finep_dualT: FINE 128 pressure -> aggregate flux to DUAL faces
%% -> DUAL-mesh transport. Source: pressure on ONE primal cell (4 fine cells), transport split over
%% the 4 corner-node dual CVs (+0.25 / -1.25 each). kappa = 5x5 tanh-blended field mapped to 64x64.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
gravity reset off
NG=64; NF=128; CFL=0.45; FPRIME_MAX=2.0; delta=0.05;
qI=1; qP=-5; Tsnaps=[0.02 0.05 0.10];
inj_reg=[0.1953125 0.3984375]; prod_reg=[0.8046875 0.5078125];   % primal 64x64 cell centers
h=1/NG; dx=[h/2, h*ones(1,NG-1), h/2]; dy=dx; xv=[0 cumsum(dx)]; yv=[0 cumsum(dy)];

%% 5x5 tanh-blended kappa, evaluated cellwise on the 64x64 PRIMAL cells, then replicated to 128 fine
Kij=[ 1.0 20.0  0.2  8.0  0.5;
      0.3  2.0 30.0  0.4 12.0;
     10.0  0.5  1.0 25.0  0.3;
      0.4 15.0  0.3  3.0 40.0;
      2.0  0.3 18.0  0.6  1.0];
edges=linspace(0,1,6); chi=@(z,a,b,d)0.5*(tanh((z-a)./d)-tanh((z-b)./d));
xcp=((1:NG)-0.5)/NG; [XP,YP]=ndgrid(xcp,xcp);              % 64 primal cell centers (i=x, j=y)
W=zeros(NG,NG); KW=zeros(NG,NG);
for i=1:5, for j=1:5
  w=chi(XP,edges(i),edges(i+1),delta).*chi(YP,edges(j),edges(j+1),delta);
  W=W+w; KW=KW+Kij(i,j)*w;
end, end
Kp=KW./max(W,1e-6);                                        % 64x64 primal kappa (cellwise constant)
kappa_fine_cell=reshape(kron(Kp,ones(2,2)),[],1);         % 128x128 (each fine cell = its primal cell)
fprintf('\n=== 5x5 blocks (delta=%.2f) + Deng inj/prod, fine-p -> dual transport ===\n',delta);
fprintf('primal kappa range [%.3f %.3f] contrast %.1fx\n',min(Kp(:)),max(Kp(:)),max(Kp(:))/min(Kp(:)));

%% fine grid + dual grid
Gf=computeGeometry(cartGrid([NF NF],[1 1])); Nf=Gf.faces.neighbors; ncf=Gf.cells.num;
rockf.perm=kappa_fine_cell; rockf.poro=ones(ncf,1);
hTf=computeTrans(Gf,rockf); Tf=1./accumarray(Gf.cells.faces(:,1),1./hTf,[Gf.faces.num,1]);
Gd=computeGeometry(tensorGrid(xv,yv)); Nd=Gd.faces.neighbors; ncd=Gd.cells.num;

%% PRIMAL source cells -> their 2x2 fine cells (pressure)
pmI=round(inj_reg*NG+0.5); pmP=round(prod_reg*NG+0.5);
fsP=@(m)[2*m-1, 2*m];
[a,b]=ndgrid(fsP(pmI(1)),fsP(pmI(2))); injFine=a(:)+(b(:)-1)*NF;
[a,b]=ndgrid(fsP(pmP(1)),fsP(pmP(2))); prodFine=a(:)+(b(:)-1)*NF;
fprintf('injector primal (%d,%d); producer primal (%d,%d)\n',pmI(1),pmI(2),pmP(1),pmP(2));

%% fine pressure solve (source on primal cells' fine cells; p=0 all boundaries)
bff=boundaryFaces(Gf); bcf=addBC([],bff,'pressure',zeros(numel(bff),1),'sat',[0 1]);
srcf=addSource([],injFine,repmat(qI/4,4,1),'sat',repmat([1 0],4,1));
srcf=addSource(srcf,prodFine,repmat(qP/4,4,1),'sat',repmat([0 1],4,1));
fluid=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
statef=incompTPFA(initResSol(Gf,0,[0 1]),Gf,Tf,fluid,'bc',bcf,'src',srcf,'use_trans',true);
fprintf('fine p range [%.4f %.4f]\n',min(statef.pressure),max(statef.pressure));

%% partition fine->dual, AGGREGATE flux (SUM fine sub-faces per dual face)
[KX,KY]=ndgrid(1:NF,1:NF); pmap=reshape(ceil((KX+1)/2)+(ceil((KY+1)/2)-1)*(NG+1),[],1);
injDualCVs=unique(pmap(injFine)); prodDualCVs=unique(pmap(prodFine));
isintf=all(Nf>0,2); di=pmap(Nf(isintf,:)); cross=di(:,1)~=di(:,2);
fint=find(isintf); fcr=fint(cross); dc=di(cross,:);
isintd=all(Nd>0,2); dfid=find(isintd);
L=sparse(min(Nd(isintd,:),[],2),max(Nd(isintd,:),[],2),dfid,ncd,ncd);
dfi=full(L(sub2ind([ncd ncd],min(dc,[],2),max(dc,[],2))));
sgn=2*(dc(:,1)==Nd(dfi,1))-1;
Qd=accumarray(dfi,statef.flux(fcr).*sgn,[Gd.faces.num,1]);
sideOf=@(c)((abs(c(:,1))<1e-9)+2*(abs(c(:,1)-1)<1e-9)+3*(abs(c(:,2))<1e-9)+4*(abs(c(:,2)-1)<1e-9));
fbf=find(~isintf); own=max(Nf(fbf,:),[],2); dcown=pmap(own);
outsgn=ones(numel(fbf),1); outsgn(Nf(fbf,1)==0)=-1; foutw=statef.flux(fbf).*outsgn;
dbf=find(~isintd); dbcown=max(Nd(~isintd,:),[],2);
Ld=sparse(dbcown,sideOf(Gd.faces.centroids(dbf,:)),dbf,ncd,4);
dbi=full(Ld(sub2ind([ncd 4],dcown,sideOf(Gf.faces.centroids(fbf,:)))));
Foutd=accumarray(dbi,foutw,[Gd.faces.num,1]);
Qb=zeros(Gd.faces.num,1); Qb(Nd(:,2)==0)=Foutd(Nd(:,2)==0); Qb(Nd(:,1)==0)=-Foutd(Nd(:,1)==0);
Qd=Qd+Qb;
divd=accumarray(Nd(Nd(:,1)>0,1),Qd(Nd(:,1)>0),[ncd,1])-accumarray(Nd(Nd(:,2)>0,2),Qd(Nd(:,2)>0),[ncd,1]);
q_dual=zeros(ncd,1); q_dual(injDualCVs)=qI/4; q_dual(prodDualCVs)=qP/4;
cons_resid=max(abs(divd-q_dual));
fprintf('AGGREGATION CHECK: max|div v_dual - q_dual| = %.3e\n',cons_resid);

%% dual-mesh transport (source split over the 4 dual CVs per well)
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

%% export (NEW name)
xc_matrix=Gd.cells.centroids;
np=Gd.faces.nodePos; aF=(1:Gd.faces.num)'; nn1=Gd.faces.nodes(np(aF)); nn2=Gd.faces.nodes(np(aF)+1);
face_p1=Gd.nodes.coords(nn1,:); face_p2=Gd.nodes.coords(nn2,:); face_centroid=Gd.faces.centroids;
face_len=Gd.faces.areas; face_normal=Gd.faces.normals./face_len; face_flux=Qd;
face_neighbors=Nd; face_is_boundary=double(any(Nd==0,2));
sw_T002=S1(:); sw_T005=S2(:); sw_T010=S3(:); snap_times=Tsnaps;
inj_cell=injDualCVs; prod_cell=prodDualCVs; inj_primal=pmI; prod_primal=pmP; q_inj=qI; q_prod=qP;
inj_fine=injFine; prod_fine=prodFine; inj_xy=[0.2 0.4]; prod_xy=[0.8 0.5];
p_matrix_fine=statef.pressure; face_flux_fine=statef.flux; kappa_primal=Kp; block_Kij=Kij; blend_delta=delta;
meta_dt=[d1 d2 d3]; meta_nsteps=[k1 k1+k2 k1+k2+k3]; meta_CFL=CFL; meta_FPRIME_MAX=FPRIME_MAX; conservation_residual=cons_resid;
meta_kappa=sprintf('5x5 deterministic tanh-blended blocks (block_Kij), delta=%.2f, mapped cellwise to 64x64 primal then replicated to 128 fine',delta);
meta_workflow='Deng homogeneous-Dirichlet inj(+1)/prod(-5): fine 128 pressure, flux summed over dual faces, dual transport; pressure source on primal cell (4 fine cells), transport split over 4 corner-node dual CVs. div v_dual=q_dual (see conservation_residual).';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_blocks5_finep_dualT.mat','-v7', ...
  'xc_matrix','p_matrix_fine','face_flux','face_neighbors','face_centroid','face_normal','face_len', ...
  'face_p1','face_p2','face_is_boundary','kappa_fine_cell','kappa_primal','block_Kij','blend_delta','face_flux_fine', ...
  'sw_T002','sw_T005','sw_T010','snap_times','inj_cell','prod_cell','inj_primal','prod_primal','inj_fine','prod_fine', ...
  'inj_xy','prod_xy','q_inj','q_prod','meta_dt','meta_nsteps','meta_CFL','meta_FPRIME_MAX','conservation_residual', ...
  'meta_kappa','meta_workflow');
fprintf('saved case3_mrst_export_blocks5_finep_dualT.mat\n');

figure('Name','blocks5 dualT log10 kappa','Position',[30 100 460 430]);
plotCellData(Gf,log10(kappa_fine_cell),'EdgeColor','none'); colormap(turbo); view(0,90); axis equal tight; colorbar;
title(sprintf('5x5 blocks log_{10}\\kappa (\\delta=%.2f)',delta)); xlabel x; ylabel y;
figure('Name','blocks5 dualT pressure','Position',[500 100 460 430]);
plotCellData(Gf,statef.pressure,'EdgeColor','none'); colormap(parula); view(0,90); axis equal tight; colorbar; hold on;
plot(inj_xy(1),inj_xy(2),'g^','MarkerSize',11,'MarkerFaceColor','g'); plot(prod_xy(1),prod_xy(2),'rv','MarkerSize',11,'MarkerFaceColor','r');
title('fine pressure p (p=0 on \partial\Omega)'); xlabel x; ylabel y;
figure('Name','blocks5 dualT dual Sw','Position',[100 60 1180 400]);
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
  fprintf('  +%.4f (%d steps): Sw_max=%.4f, cells S>0.5=%d\n',Tdur,nsteps,max(S),nnz(S>0.5));
end
