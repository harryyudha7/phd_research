%% Heterogeneous tanh permeability (nonfractured). kappa smooth, tunable sharpness.
%% kappa(x,y) = 1 + (Khi-1)*0.5*(1+tanh((x+y-1)/delta))  -> diagonal interface x+y=1.
%% f=0, p=1 (x=1) -> p=0 (x=0), no-flow top/bottom. Standard MRST = harmonic averaging.
%% Also: delta-sweep comparing harmonic-average vs evaluate-kappa-at-face transmissibility.
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
gravity reset off

Khi=10; delta=0.1;   % more diffuse interface
kfun=@(XY,d) 1 + (Khi-1)*0.5.*(1+tanh((XY(:,1)+XY(:,2)-1)/d));

G=computeGeometry(cartGrid([128 128],[1 1]));
xc=G.cells.centroids; N=G.faces.neighbors;
rock.perm=kfun(xc,delta); rock.poro=ones(G.cells.num,1); G.rock=rock;

%% standard MRST solve (harmonic averaging in computeTrans)
hT=computeTrans(G,rock);
T_harm=1./accumarray(G.cells.faces(:,1),1./hT,[G.faces.num,1]);   % full face trans (harmonic)
bc=pside(pside([],G,'LEFT',1),G,'RIGHT',4);   % Koppel-Martin: p=1 (x=0), p=4 (x=1)
state=incompTPFA(initResSol(G,0),G,T_harm,initSingleFluid('mu',1,'rho',1),'bc',bc,'use_trans',true);
% conservation check (f=0 -> div v = 0)
divv=accumarray(N(N(:,1)>0,1),state.flux(N(:,1)>0),[G.cells.num,1]) ...
    -accumarray(N(N(:,2)>0,2),state.flux(N(:,2)>0),[G.cells.num,1]);
fprintf('\n=== tanh-perm solve (harmonic, delta=%.3f, contrast %dx) ===\n',delta,Khi);
fprintf('p range [%.4f %.4f], max|div v| = %.3e (should be ~machine)\n',min(state.pressure),max(state.pressure),max(abs(divv)));

%% delta-sweep: harmonic-mean(kappa_i,kappa_j) vs kappa(face) per interior face
isInt=N(:,1)>0 & N(:,2)>0; fi=find(isInt);
fc=G.faces.centroids(fi,:);
fprintf('\n=== harmonic vs face-evaluation transmissibility (interior faces) ===\n');
fprintf('  delta     max relerr   mean relerr   %%faces>1%%\n');
for d=[0.10 0.05 0.02 0.01 0.005]
    kc=kfun(xc,d); ki=kc(N(fi,1)); kj=kc(N(fi,2));
    Th=2*ki.*kj./(ki+kj);            % harmonic mean (what MRST uses)
    Tf=kfun(fc,d);                   % evaluate kappa at the face
    rel=abs(Th-Tf)./Tf;
    fprintf('  %.3f     %.3e    %.3e     %.1f%%\n',d,max(rel),mean(rel),100*mean(rel>0.01));
end

%% flux difference: harmonic vs face-eval solve, at delta=0.05 and 0.01
for d=[0.05 0.01]
    rock.perm=kfun(xc,d);
    hT=computeTrans(G,rock); Th_full=1./accumarray(G.cells.faces(:,1),1./hT,[G.faces.num,1]);
    Tf_full=Th_full; Tf_full(fi)=kfun(fc,d).*G.faces.areas(fi)./vecnorm(xc(N(fi,1),:)-xc(N(fi,2),:),2,2);
    sH=incompTPFA(initResSol(G,0),G,Th_full,initSingleFluid('mu',1,'rho',1),'bc',bc,'use_trans',true);
    sF=incompTPFA(initResSol(G,0),G,Tf_full,initSingleFluid('mu',1,'rho',1),'bc',bc,'use_trans',true);
    rf=norm(sH.flux-sF.flux)/norm(sH.flux); rp=norm(sH.pressure-sF.pressure)/norm(sH.pressure);
    fprintf('delta=%.3f : harmonic-vs-faceeval  flux relL2=%.3e  pressure relL2=%.3e\n',d,rf,rp);
end

%% export (delta=0.05 harmonic solve) + plots
rock.perm=kfun(xc,delta);
kappa_cell=rock.perm; p_matrix=state.pressure; xc_matrix=xc;
np=G.faces.nodePos; aF=(1:G.faces.num)'; n1=G.faces.nodes(np(aF)); n2=G.faces.nodes(np(aF)+1);
face_p1=G.nodes.coords(n1,:); face_p2=G.nodes.coords(n2,:); face_centroid=G.faces.centroids;
face_len=G.faces.areas; face_normal=G.faces.normals./face_len; face_flux=state.flux;
face_neighbors=N; face_is_boundary=double(any(N==0,2));
meta_kappa='kappa=1+(Khi-1)*0.5*(1+tanh((x+y-1)/delta))'; meta_Khi=Khi; meta_delta=delta;
meta_bc='Koppel-Martin matrix BCs: p=1 (x=0), p=4 (x=1), no-flow top/bottom; f=0; NO fracture'; meta_avg='MRST harmonic averaging (computeTrans)';
README='Heterogeneous tanh-perm nonfractured case. kappa_cell=permeability; face_flux=MRST conservative flux (harmonic avg). For CG: use kappa=1+(Khi-1)*0.5*(1+tanh((x+y-1)/delta)).';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_tanh_d0p1.mat','-v7', ...
  'kappa_cell','p_matrix','xc_matrix','face_p1','face_p2','face_centroid','face_normal','face_len', ...
  'face_flux','face_neighbors','face_is_boundary','meta_kappa','meta_Khi','meta_delta','meta_bc','meta_avg','README');
fprintf('\nsaved case3_mrst_export_tanh_d0p1.mat\n');

figure('Name','tanh kappa','Position',[60 80 520 460]);
plotCellData(G,kappa_cell,'EdgeColor','none'); colormap(parula); view(0,90); axis equal tight; colorbar;
title(sprintf('\\kappa (tanh, \\delta=%.3f, %dx)',delta,Khi)); xlabel x; ylabel y;
figure('Name','tanh pressure','Position',[600 80 520 460]);
plotCellData(G,state.pressure,'EdgeColor','none'); colormap(jet); view(0,90); axis equal tight; colorbar;
hold on; line([0 1],[1 0],'Color','k','LineStyle','--');   % interface x+y=1
title('pressure (flow refracts at \kappa interface)'); xlabel x; ylabel y;
