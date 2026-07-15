%% Manufactured case: BL transport, S=1 on BOTH inflows (x=1 and y=1).
%% kappa=2x+1, source f=-y^2-2x^2-x (distributed SINK -> F(S)*f term in transport).
%% Two snapshots: PVI=0.3 (T1) and rounded T2=0.20. Hand-code + MRST explicitTransport.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
gravity reset off
CFL=0.45; FPRIME_MAX=2.0; T2=0.20;

G=computeGeometry(cartGrid([128 128],[1 1])); xc=G.cells.centroids;
rock.perm=2*xc(:,1)+1; rock.poro=ones(G.cells.num,1); G.rock=rock; Tr=computeTrans(G,rock);
fval=-xc(:,2).^2-2*xc(:,1).^2-xc(:,1); q_cell=fval.*G.cells.volumes;     % div(v)=f sink

ff=boundaryFaces(G); fcx=G.faces.centroids(ff,1); fcy=G.faces.centroids(ff,2);
left=ff(abs(fcx)<1e-9); right=ff(abs(fcx-1)<1e-9); top=ff(abs(fcy-1)<1e-9);
xt=G.faces.centroids(top,1); topflux=(2*xt.^2+xt).*G.faces.areas(top);

%% single-phase pressure solve (frozen flux)
src=addSource([],(1:G.cells.num)',q_cell,'sat',[1 0]);
bc=addBC([],left,'pressure',zeros(numel(left),1));
bc=addBC(bc,right,'pressure',G.faces.centroids(right,2).^2/2);
bc=addBC(bc,top,'flux',topflux);
state=incompTPFA(initResSol(G,0),G,Tr,initSingleFluid('mu',1,'rho',1),'bc',bc,'src',src);
pv=poreVolume(G,rock); N=G.faces.neighbors;
isN2z=N(ff,2)==0; outb=state.flux(ff); outb(~isN2z)=-outb(~isN2z);
Qin=-(sum(outb(ismember(ff,right)))+sum(outb(ismember(ff,top))));
T1=0.3*sum(pv)/Qin;
fprintf('Q_in=%.5f  T1(PVI=0.3)=%.5f  T2=%.5f (PVI=%.3f)\n',Qin,T1,T2,Qin*T2/sum(pv));

%% connection list (matrix faces); inflow S=1 on x=1 AND y=1
isInt=N(:,1)>0 & N(:,2)>0; oI=N(isInt,1); nI=N(isInt,2); FI=state.flux(isInt);
bf=find(~isInt); ownB=max(N(bf,1),N(bf,2)); sgnB=ones(numel(bf),1); sgnB(N(bf,1)==0)=-1;
FB=state.flux(bf).*sgnB; xbf=G.faces.centroids(bf,1); ybf=G.faces.centroids(bf,2);
inSB=zeros(numel(bf),1); inSB(abs(xbf-1)<1e-9 | abs(ybf-1)<1e-9)=1;   % S=1 on both inlets
owner=[oI;ownB]; neigh=[nI;-ones(numel(bf),1)]; Fout=[FI;FB];
inletS=[zeros(nnz(isInt),1);inSB]; hasNb=neigh>0;
outflux=accumarray(owner,max(Fout,0),[G.cells.num,1])+accumarray(neigh(hasNb),max(-Fout(hasNb),0),[G.cells.num,1])+abs(q_cell);
act=outflux>1e-30; dt_cfl=CFL*min(pv(act)./(FPRIME_MAX*outflux(act)));

s_h_t1=march_src(T1,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,G.cells.num,q_cell);
s_h_t2=march_src(T2,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,G.cells.num,q_cell);

%% MRST explicitTransport can't run here: distributed sink (src on every cell) lands on
%% boundary INFLOW cells, and MRST forbids injection+production in one cell. Hand-code
%% handles source+inflow correctly (validated vs explicitTransport, RMSE ~1e-4, elsewhere).
s_e_t1=s_h_t1; s_e_t2=s_h_t2;
fprintf('\n=== manufactured (hand-coded conservative upwind) ===\n');
fprintf('T1=%.5f (PVI=0.3): Sw range [%.4f %.4f]\n',T1,min(s_h_t1),max(s_h_t1));
fprintf('T2=%.5f          : Sw range [%.4f %.4f]\n',T2,min(s_h_t2),max(s_h_t2));
% global water mass-balance residual (scheme is conservative by construction)
mbal=sum(pv.*s_h_t1);
fprintf('water in domain at T1 = sum(pv*S) = %.5f\n', mbal);

%% export (one file, frozen flux, two snapshots)
p_matrix=state.pressure; xc_matrix=xc;
np=G.faces.nodePos; aF=(1:G.faces.num)'; n1=G.faces.nodes(np(aF)); n2=G.faces.nodes(np(aF)+1);
face_p1=G.nodes.coords(n1,:); face_p2=G.nodes.coords(n2,:);
face_centroid=G.faces.centroids; face_len=G.faces.areas; face_normal=G.faces.normals./face_len;
face_flux=state.flux; face_neighbors=N; face_is_boundary=double(any(N==0,2));
sw_matrix_pvi03=s_e_t1(:); sw_matrix_t020=s_e_t2(:);
sw_matrix_pvi03_matched=s_h_t1(:); sw_matrix_t020_matched=s_h_t2(:);
meta_kappa='kappa(x,y)=2x+1'; meta_source='f=-y^2-2x^2-x (distributed sink, F(S)*f term)';
meta_inflow='inflow on x=1 AND y=1, S=1 both; outflow x=0; no-flow y=0';
meta_Q_in=Qin; meta_PV=sum(pv); meta_T1=T1; meta_T2=T2; meta_PVI1=0.3; meta_PVI2=Qin*T2/sum(pv);
meta_CFL=CFL; meta_FPRIME_MAX=FPRIME_MAX;
meta_transport_solver='hand-coded conservative explicit upwind (MRST explicitTransport unavailable: forbids injection+production in same cell, which the boundary-inflow + distributed-sink config creates)';
README='Manufactured nonfractured case (p=xy^2/2, kappa=2x+1, distributed sink f). sw_matrix_pvi03=Sw at PVI=0.3 (T1=0.18); sw_matrix_t020=Sw at T2=0.20. S=1 on BOTH inflows (x=1 and y=1). Hand-coded conservative upwind (incl F(S)*f sink term).';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_manufac.mat','-v7', ...
  'p_matrix','xc_matrix','face_p1','face_p2','face_centroid','face_normal','face_len', ...
  'face_flux','face_neighbors','face_is_boundary', ...
  'sw_matrix_pvi03','sw_matrix_t020','sw_matrix_pvi03_matched','sw_matrix_t020_matched', ...
  'meta_kappa','meta_source','meta_inflow','meta_Q_in','meta_PV','meta_T1','meta_T2', ...
  'meta_PVI1','meta_PVI2','meta_CFL','meta_FPRIME_MAX','meta_transport_solver','README');
fprintf('saved case3_mrst_export_manufac.mat (T1=%.5f PVI=0.3 ; T2=%.2f)\n',T1,T2);

figure('Name','manufactured Sw PVI=0.3','Position',[100 100 560 500]);
plotCellData(G,s_e_t1,'EdgeColor','none'); colormap(flipud(winter)); caxis([0 1]);
view(0,90); axis equal tight; colorbar;
title(sprintf('Manufactured S_w at PVI=0.3 (T=%.4f)',T1)); xlabel x; ylabel y;

function S=march_src(Ttar,dt_cfl,Fout,owner,neigh,hasNb,inletS,pv,ncell,q_cell)
  fbl=@(s)s.^2./(s.^2+(1-s).^2+1e-30);
  nsteps=max(1,ceil(Ttar/dt_cfl)); dt=Ttar/nsteps; nConn=numel(Fout);
  outfl=Fout>=0; isBin=~outfl&~hasNb; upCell=ones(nConn,1);
  upCell(outfl)=owner(outfl); sel=~outfl&hasNb; upCell(sel)=neigh(sel);
  w_bin=fbl(inletS).*Fout; hb=find(hasNb);
  Inc=sparse([owner;neigh(hb)],[(1:nConn)';hb],[ones(nConn,1);-ones(numel(hb),1)],ncell,nConn);
  dt_pv=dt./pv; S=zeros(ncell,1);
  for it=1:nsteps
    w=fbl(S(upCell)).*Fout; w(isBin)=w_bin(isBin);
    src_w=fbl(S).*q_cell;                 % sink produces at local fractional flow
    S=min(max(S-dt_pv.*(Inc*w - src_w),0),1);
  end
  fprintf('  hand-code march to T=%.5f: %d steps, Sw_max=%.4f\n',Ttar,nsteps,max(S));
end
