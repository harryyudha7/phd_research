%% Q2 dual transport: SPE10 L20 kappa + Deng inj/prod, ONE LEVEL FINER than the Q1 case.
%% Pressure: 256x256 TPFA (16 fine cells per 64x64 primal cell). Transport: 129x129 Q2 dual CVs.
%% Q2 dual CV edges = midpoints of the 129 Q2 nodes (+0,1) -> each dual CV = union of fine cells.
%% Source: pressure split over 16 fine cells/primal; transport = tensor weights [1/4,1/2,1/4]^2 over 9 nodal CVs
%%         (obtained by aggregating the fine-cell source -> identical, and = div v_dual by construction).
%% Injector S=1; producer removes at local F(S); boundary inflow S=0. Writes case3_mrst_export_spe10_q2_dualT.mat.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add spe10 incomp
gravity reset off
LAYER=20; NG=64; NF=256; NQ=129; CFL=0.45; FPRIME_MAX=2.0;
qI=1; qP=-5; Tsnaps=[0.02 0.05 0.10 0.25 0.50 0.80];
inj_reg=[0.1953125 0.3984375]; prod_reg=[0.8046875 0.5078125];

%% Q2 dual grid (129x129): edges at Q2-node midpoints, half cells at the outer boundary
xq=linspace(0,1,NQ); yq=linspace(0,1,NQ);
xe=[0, 0.5*(xq(1:end-1)+xq(2:end)), 1]; ye=[0, 0.5*(yq(1:end-1)+yq(2:end)), 1];
Gd=computeGeometry(tensorGrid(xe,ye)); Nd=Gd.faces.neighbors; ncd=Gd.cells.num;

%% SPE10 L20 kappa: resample 60x220 -> 64x64 primal (nearest, geomean=1), then replicate 4x4 -> 256x256 fine
rk=getSPE10rock(LAYER); Kx=rk.perm(:,1); kappa=Kx./exp(mean(log(Kx)));
Korig=reshape(kappa,[60 220]);
ispe=min(60,max(1,ceil(((1:NG)-0.5)/NG*60))); jspe=min(220,max(1,ceil(((1:NG)-0.5)/NG*220)));
Kp=Korig(ispe,jspe); kappa_fine_cell=reshape(kron(Kp,ones(4,4)),[],1);
fprintf('\n=== SPE10 L%d Q2-dual (fine 256, dual 129) ===\n',LAYER);
fprintf('primal kappa range [%.3e %.3e] contrast %.2e x\n',min(Kp(:)),max(Kp(:)),max(Kp(:))/min(Kp(:)));

Gf=computeGeometry(cartGrid([NF NF],[1 1])); Nf=Gf.faces.neighbors; ncf=Gf.cells.num;
rockf.perm=kappa_fine_cell; rockf.poro=ones(ncf,1);
hTf=computeTrans(Gf,rockf); Tf=1./accumarray(Gf.cells.faces(:,1),1./hTf,[Gf.faces.num,1]);

%% primal source cells -> their 4x4=16 fine cells (pressure source +1/16, -5/16)
pmI=round(inj_reg*NG+0.5); pmP=round(prod_reg*NG+0.5);
fsP4=@(m)(4*m-3):(4*m);
[a,b]=ndgrid(fsP4(pmI(1)),fsP4(pmI(2))); injFine=a(:)+(b(:)-1)*NF;
[a,b]=ndgrid(fsP4(pmP(1)),fsP4(pmP(2))); prodFine=a(:)+(b(:)-1)*NF;
bff=boundaryFaces(Gf); bcf=addBC([],bff,'pressure',zeros(numel(bff),1),'sat',[0 1]);
srcf=addSource([],injFine,repmat(qI/16,16,1),'sat',repmat([1 0],16,1));
srcf=addSource(srcf,prodFine,repmat(qP/16,16,1),'sat',repmat([0 1],16,1));
fluid=initSimpleFluid('mu',[1 1],'rho',[1 1],'n',[2 2]);
fprintf('solving 256x256 pressure...\n');
statef=incompTPFA(initResSol(Gf,0,[0 1]),Gf,Tf,fluid,'bc',bcf,'src',srcf,'use_trans',true);
fprintf('fine p range [%.4f %.4f]\n',min(statef.pressure),max(statef.pressure));

%% partition fine(256)->Q2 dual(129), aggregate flux, and dual_source_rate
[KX,KY]=ndgrid(1:NF,1:NF); pmap=reshape(ceil((KX+1)/2)+(ceil((KY+1)/2)-1)*NQ,[],1);
q_fine=zeros(ncf,1); q_fine(injFine)=qI/16; q_fine(prodFine)=qP/16;
dual_source_rate=accumarray(pmap,q_fine,[ncd,1]);
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

%% VALIDATION
divd=accumarray(Nd(Nd(:,1)>0,1),Qd(Nd(:,1)>0),[ncd,1])-accumarray(Nd(Nd(:,2)>0,2),Qd(Nd(:,2)>0),[ncd,1]);
cons_resid=max(abs(divd-dual_source_rate));
fprintf('CHECK max|div v_dual - dual_source_rate| = %.3e (~1e-12)\n',cons_resid);
fprintf('CHECK sum(dual_source_rate)      = %.10f (want -4)\n',sum(dual_source_rate));
fprintf('CHECK sum(max(rate,0))           = %.10f (want  1)\n',sum(max(dual_source_rate,0)));
fprintf('CHECK sum(min(rate,0))           = %.10f (want -5)\n',sum(min(dual_source_rate,0)));
fprintf('CHECK sum(dual_cell_area)        = %.10f (want  1)\n',sum(Gd.cells.volumes));

%% transport on the Q2 dual grid
isIntd=all(Nd>0,2); oI=Nd(isIntd,1); nI=Nd(isIntd,2); FI=Qd(isIntd);
bfd=find(~isIntd); ownB=max(Nd(bfd,1),Nd(bfd,2)); sgnB=ones(numel(bfd),1); sgnB(Nd(bfd,1)==0)=-1; FB=Qd(bfd).*sgnB;
owner=[oI;ownB]; neigh=[nI;-ones(numel(bfd),1)]; Fout=[FI;FB]; hasNb=neigh>0;
pvd=Gd.cells.volumes;
outflux=accumarray(owner,max(Fout,0),[ncd,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[ncd,1])+abs(dual_source_rate);
act=outflux>1e-30; dt_cfl=CFL*min(pvd(act)./(FPRIME_MAX*outflux(act)));
posSrc=dual_source_rate>0; negSrc=dual_source_rate<0;
fprintf('dual PV=%.5f, dt_cfl=%.3e\n',sum(pvd),dt_cfl);
SS=cell(1,6); nst=zeros(1,6); dts=zeros(1,6); S=zeros(ncd,1); tprev=0;
for s=1:6
  [S,nst(s),dts(s)]=march_q2(S,Tsnaps(s)-tprev,dt_cfl,Fout,owner,neigh,hasNb,pvd,ncd,dual_source_rate,posSrc,negSrc);
  SS{s}=S; tprev=Tsnaps(s);
end

%% export
xc_matrix=Gd.cells.centroids;
np=Gd.faces.nodePos; aF=(1:Gd.faces.num)'; nn1=Gd.faces.nodes(np(aF)); nn2=Gd.faces.nodes(np(aF)+1);
face_p1=Gd.nodes.coords(nn1,:); face_p2=Gd.nodes.coords(nn2,:); face_centroid=Gd.faces.centroids;
face_len=Gd.faces.areas; face_normal=Gd.faces.normals./face_len; face_flux=Qd;
face_neighbors=Nd; face_is_boundary=double(any(Nd==0,2));
sw_T002=SS{1}(:); sw_T005=SS{2}(:); sw_T010=SS{3}(:); sw_T025=SS{4}(:); sw_T050=SS{5}(:); sw_T080=SS{6}(:);
snap_times=Tsnaps; meta_dt=dts; meta_nsteps=cumsum(nst); meta_CFL=CFL; meta_FPRIME_MAX=FPRIME_MAX;
meta_celldim=[NQ NQ]; meta_pressure_celldim=[NF NF]; meta_transport_grid='Q2_dual'; conservation_residual=cons_resid;
inj_primal=pmI; prod_primal=pmP; q_inj=qI; q_prod=qP; inj_xy=[0.2 0.4]; prod_xy=[0.8 0.5];
p_matrix_fine=statef.pressure; kappa_primal=Kp;
meta_source=sprintf('SPE10 model-2 layer %d, Kx normalized geomean=1, resampled 60x220 -> 64x64 nearest, replicated 4x4 -> 256x256',LAYER);
meta_workflow='Q2 dual transport. Pressure 256x256 TPFA (src +1/16,-5/16 over 16 fine cells/primal). Flux summed onto 129x129 Q2 dual faces. Transport source = tensor [1/4,1/2,1/4]^2 over 9 nodal CVs (= aggregated fine src = div v_dual). Injector S=1, producer local F(S), boundary S=0.';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_spe10_q2_dualT.mat','-v7', ...
  'face_flux','face_neighbors','face_centroid','face_normal','face_len','face_p1','face_p2','xc_matrix', ...
  'dual_source_rate','sw_T002','sw_T005','sw_T010','sw_T025','sw_T050','sw_T080','snap_times', ...
  'meta_dt','meta_nsteps','meta_celldim','meta_pressure_celldim','meta_transport_grid','meta_CFL','meta_FPRIME_MAX', ...
  'conservation_residual','inj_primal','prod_primal','q_inj','q_prod','inj_xy','prod_xy', ...
  'p_matrix_fine','kappa_primal','meta_source','meta_workflow');
fprintf('saved case3_mrst_export_spe10_q2_dualT.mat\n');

%% figures
figure('Name','spe10 q2 fine log10 kappa','Position',[30 100 460 430]);
plotCellData(Gf,log10(kappa_fine_cell),'EdgeColor','none'); colormap(jet); view(0,90); axis equal tight; colorbar;
title('SPE10 L20 256^2 fine log_{10}\kappa'); xlabel x; ylabel y;
figure('Name','spe10 q2 dual Sw','Position',[60 40 1240 660]);
for k=1:6
  subplot(2,3,k);
  plotCellData(Gd,SS{k},'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]); view(0,90); axis equal tight; hold on;
  plot(inj_xy(1),inj_xy(2),'g^','MarkerSize',9,'MarkerFaceColor','g'); plot(prod_xy(1),prod_xy(2),'rv','MarkerSize',9,'MarkerFaceColor','r');
  lab=''; if k==2, lab=' (selected)'; end
  title(sprintf('Q2 dual S_w at T=%.2f%s',Tsnaps(k),lab)); xlabel x; ylabel y;
end
colorbar('Position',[0.94 0.15 0.012 0.7]);

function [S,nsteps,dt]=march_q2(S,Tdur,dt_cfl,Fout,owner,neigh,hasNb,pv,ncell,qsrc,pos,neg)
  fbl=@(s)s.^2./(s.^2+(1-s).^2+1e-30);
  nsteps=max(1,ceil(Tdur/dt_cfl)); dt=Tdur/nsteps; nConn=numel(Fout);
  outfl=Fout>=0; isBin=~outfl&~hasNb; upCell=ones(nConn,1);
  upCell(outfl)=owner(outfl); sel=~outfl&hasNb; upCell(sel)=neigh(sel);
  hb=find(hasNb);
  Inc=sparse([owner;neigh(hb)],[(1:nConn)';hb],[ones(nConn,1);-ones(numel(hb),1)],ncell,nConn);
  dt_pv=dt./pv;
  for it=1:nsteps
    w=fbl(S(upCell)).*Fout; w(isBin)=0;
    A=Inc*w; qw=zeros(ncell,1); qw(pos)=qsrc(pos); qw(neg)=qsrc(neg).*fbl(S(neg));
    S=min(max(S+dt_pv.*(-A+qw),0),1);
  end
  fprintf('  +%.4f (%d steps): Sw_max=%.4f, cells S>0.5=%d\n',Tdur,nsteps,max(S),nnz(S>0.5));
end
