%% Case-1 EDFM sealed-tip v2 -- DUAL-MESH transport: aggregate 128 matrix flux -> 65x65 nodal dual,
%% retain the 125 MRST fracture cells (native), aggregate the NNC matrix side onto the dual cells,
%% run the same explicit-upwind transport on the 65-dual-matrix + 125-fracture grid (coincides with CG).
%% Adds dual fields to case3_mrst_export_noflow_v2.mat (option a): keeps the 128-native matrix face_flux.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
ckpt='c:\Users\muchamad\mrst-project\case3_noflow_v2_checkpoint.mat';
outfile='c:\Users\muchamad\mrst-project\case3_mrst_export_noflow_v2.mat';
load(ckpt);   % G,state,N,pv,nc,nfrac,ellF, face_flux(matrix),face_neighbors,face_centroid, frac_face_flux,frac_face_neighbors, mfMatCell,mfFracCell,m2f, Qright,T_final,RIGHT_BC_S,CFL,FPRIME_MAX,...
NGm=128; NGp=64; NGd=NGp+1;   % matrix 128, CG primal 64, nodal dual 65

%% 65x65 nodal-dual matrix grid (same as SPE10 finep_dualT dual)
hp=1/NGp; dx=[hp/2, hp*ones(1,NGp-1), hp/2]; xv=[0 cumsum(dx)];
Gd=computeGeometry(tensorGrid(xv,xv)); Nd=Gd.faces.neighbors; ncd=Gd.cells.num;   % 4225
ncomb=ncd+nfrac;   % 4225 + 125 = 4350
% partition: 128 matrix cell (kx,ky) -> 65 dual cell
[KX,KY]=ndgrid(1:NGm,1:NGm); pmap=reshape(ceil((KX+1)/2)+(ceil((KY+1)/2)-1)*NGd,[],1);   % 16384 -> 4225

%% aggregate 128 matrix faces -> 65 dual faces (sum, MRST-native sign)
Nm=face_neighbors; fluxM=face_flux; cenM=face_centroid;      % matrix faces (128 cell idx or 0)
isintM=all(Nm>0,2); di=pmap(Nm(isintM,:)); cross=di(:,1)~=di(:,2);
fintM=find(isintM); fcr=fintM(cross); dc=di(cross,:);
isintd=all(Nd>0,2); dfid=find(isintd);
L=sparse(min(Nd(isintd,:),[],2),max(Nd(isintd,:),[],2),dfid,ncd,ncd);
dfi=full(L(sub2ind([ncd ncd],min(dc,[],2),max(dc,[],2))));
sgn=2*(dc(:,1)==Nd(dfi,1))-1;
Qd=accumarray(dfi,fluxM(fcr).*sgn,[Gd.faces.num,1]);
% boundary matrix faces -> dual boundary faces
sideOf=@(c)((abs(c(:,1))<1e-9)+2*(abs(c(:,1)-1)<1e-9)+3*(abs(c(:,2))<1e-9)+4*(abs(c(:,2)-1)<1e-9));
fbM=find(~isintM); ownB=max(Nm(fbM,:),[],2); dcown=pmap(ownB);
outs=ones(numel(fbM),1); outs(Nm(fbM,1)==0)=-1; foutw=fluxM(fbM).*outs;   % >0 = leaving domain
dbf=find(~isintd); dbcown=max(Nd(~isintd,:),[],2);
Ld=sparse(dbcown,sideOf(Gd.faces.centroids(dbf,:)),dbf,ncd,4);
dbi=full(Ld(sub2ind([ncd 4],dcown,sideOf(cenM(fbM,:)))));
Foutd=accumarray(dbi,foutw,[Gd.faces.num,1]);
Qb=zeros(Gd.faces.num,1); Qb(Nd(:,2)==0)=Foutd(Nd(:,2)==0); Qb(Nd(:,1)==0)=-Foutd(Nd(:,1)==0);
Qd=Qd+Qb;
% conservation of the aggregated matrix flux (before adding NNC/fracture): div should equal -(net NNC into each dual cell)
fprintf('aggregated dual matrix flux built (%d dual faces)\n',Gd.faces.num);

%% aggregate NNC matrix side onto dual cells: (dual matrix cell, fracture-local) -> summed m2f
fracLocal=mfFracCell-nc;                         % 1..125
key=[pmap(mfMatCell) fracLocal];
[ukey,~,ic]=unique(key,'rows'); m2f_agg=accumarray(ic,m2f);
nnc_dualmat=ukey(:,1); nnc_fraclocal=ukey(:,2);
fprintf('NNC aggregated: %d raw -> %d (dual matrix cell, fracture) connections\n',numel(m2f),numel(m2f_agg));

%% combined connection list on [65-dual matrix (1..ncd)] + [125 fracture (ncd+1..ncomb)]
% (1) matrix dual faces (internal + boundary), inlet S=1 at x=1
isIntd=all(Nd>0,2); oI=Nd(isIntd,1); nI=Nd(isIntd,2); FI=Qd(isIntd);
bfd=find(~isIntd); ownBd=max(Nd(bfd,1),Nd(bfd,2)); sgd=ones(numel(bfd),1); sgd(Nd(bfd,1)==0)=-1; FBd=Qd(bfd).*sgd;
xbd=Gd.faces.centroids(bfd,1); inS_b=zeros(numel(bfd),1); inS_b(abs(xbd-1)<1e-9)=RIGHT_BC_S;
% (2) fracture-internal faces (remap nc+i -> ncd+i)
ff1=frac_face_neighbors(:,1)-nc+ncd; ff2=frac_face_neighbors(:,2)-nc+ncd; Fff=frac_face_flux;
% (3) NNC (dual matrix cell -> fracture cell), m2f>0 = matrix into fracture
nncOwn=nnc_dualmat; nncNb=ncd+nnc_fraclocal; nncF=m2f_agg;
owner =[oI; ownBd;               ff1; nncOwn];
neigh =[nI; -ones(numel(bfd),1); ff2; nncNb];
Fout  =[FI; FBd;                 Fff; nncF];
inletS=[zeros(nnz(isIntd),1); inS_b; zeros(numel(Fff),1); zeros(numel(nncF),1)];
hasNb=neigh>0;
pv_comb=[Gd.cells.volumes; pv(nc+1:end)];        % dual matrix volumes + fracture pore volumes (phi=1)

%% conservation check: steady div(flux) per combined cell must vanish
Incx=sparse([owner;neigh(hasNb)],[(1:numel(Fout))';find(hasNb)],[ones(numel(Fout),1);-ones(nnz(hasNb),1)],ncomb,numel(Fout));
divc=Incx*Fout; fprintf('CHECK: max|div v| over %d combined cells = %.3e (should ~solver tol)\n',ncomb,max(abs(divc)));
% right-boundary inflow preserved?
fprintf('CHECK: aggregated x=1 inflow = %.4f (native Q_water=%.4f)\n',sum(abs(FBd(abs(xbd-1)<1e-9))),Qright);

%% explicit upwind transport (same scheme), 10 snapshots to PVI=1.0
outflux=accumarray(owner,max(Fout,0),[ncomb,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[ncomb,1]);
act=outflux>1e-30; dt_cfl=CFL*min(pv_comb(act)./(FPRIME_MAX*outflux(act)));
fbl=@(s)s.^2./(s.^2+(1-s).^2+1e-30); nConn=numel(Fout);
outfl=Fout>=0; isBin=~outfl&~hasNb; upCell=ones(nConn,1);
upCell(outfl)=owner(outfl); sel=~outfl&hasNb; upCell(sel)=neigh(sel);
w_bin=fbl(inletS).*Fout; hb=find(hasNb);
Inc=sparse([owner;neigh(hb)],[(1:nConn)';hb],[ones(nConn,1);-ones(numel(hb),1)],ncomb,nConn);
snap_PVI=(1:10)/10; snap_T_abs=snap_PVI*T_final;
sw_matrix_dual_snaps=zeros(ncd,10); sw_frac_snaps=zeros(nfrac,10);
fprintf('dual transport: dt_cfl=%.3e, 10 snapshots to PVI=1 (T=%.5f)\n',dt_cfl,T_final);
S=zeros(ncomb,1); tprev=0;
for sn=1:10
  Ttar=snap_T_abs(sn); ns=max(1,ceil((Ttar-tprev)/dt_cfl)); dt=(Ttar-tprev)/ns; dt_pv=dt./pv_comb;
  for it=1:ns
    w=fbl(S(upCell)).*Fout; w(isBin)=w_bin(isBin);
    S=min(max(S-dt_pv.*(Inc*w),0),1);
  end
  sw_matrix_dual_snaps(:,sn)=S(1:ncd); sw_frac_snaps(:,sn)=S(ncd+1:end); tprev=Ttar;
  fprintf('  snap %2d/10 PVI=%.1f T=%.5f (%d steps) matrix Sw_max=%.4f frac Sw_max=%.4f\n',sn,snap_PVI(sn),Ttar,ns,max(S(1:ncd)),max(S(ncd+1:end)));
end

%% ===== add dual fields to the v2 export (option a: keep 128-native, add dual) =====
xc_matrix_dual=Gd.cells.centroids; face_flux_dual=Qd; face_neighbors_dual=Nd;
face_centroid_dual=Gd.faces.centroids; face_len_dual=Gd.faces.areas; face_normal_dual=Gd.faces.normals./face_len_dual;
np=Gd.faces.nodePos; aF=(1:Gd.faces.num)'; n1=Gd.faces.nodes(np(aF)); n2=Gd.faces.nodes(np(aF)+1);
face_p1_dual=Gd.nodes.coords(n1,:); face_p2_dual=Gd.nodes.coords(n2,:); face_is_boundary_dual=double(any(Nd==0,2));
nnc_dualmat_cell=nnc_dualmat; nnc_frac_local=nnc_fraclocal; nnc_flux_m2f_dual=m2f_agg;
sw_matrix_snaps=sw_matrix_dual_snaps;   % transport IS the dual matrix now (was 128-native)
sw_matrix=sw_matrix_dual_snaps(:,10); sw_frac=sw_frac_snaps(:,10);
meta_dualdim=[NGd NGd]; meta_transport_grid='65x65 nodal dual matrix + 125 native MRST fracture cells';
meta_dual_massbal=max(abs(divc)); meta_dual_inflow=sum(abs(FBd(abs(xbd-1)<1e-9)));
meta_transport_solver='explicit first-order upwind on 65x65 dual matrix (aggregated MRST flux) + 125 fracture (native flux + aggregated NNC); F(S)=S^2/(S^2+(1-S)^2); S=1 on x=1 dual boundary';
conventions.dual_transport='matrix transport on 65x65 nodal dual (128 matrix flux summed onto dual faces); fracture retains 125 native MRST cells; NNC matrix side aggregated to dual cell. Native 128 matrix face_flux kept for reference. sw_matrix_snaps is now (ncd x 10) on the dual.';
save(outfile,'-append','xc_matrix_dual','face_flux_dual','face_neighbors_dual','face_centroid_dual','face_len_dual', ...
  'face_normal_dual','face_p1_dual','face_p2_dual','face_is_boundary_dual','nnc_dualmat_cell','nnc_frac_local','nnc_flux_m2f_dual', ...
  'sw_matrix_dual_snaps','sw_matrix_snaps','sw_frac_snaps','sw_matrix','sw_frac','snap_PVI','snap_T_abs', ...
  'meta_dualdim','meta_transport_grid','meta_dual_massbal','meta_dual_inflow','meta_transport_solver','conventions');
fprintf('appended dual-mesh fields to %s\n',outfile);
fprintf('dual matrix Sw range [%.4f %.4f], fracture Sw range [%.4f %.4f] at PVI=1.0\n',min(sw_matrix),max(sw_matrix),min(sw_frac),max(sw_frac));
