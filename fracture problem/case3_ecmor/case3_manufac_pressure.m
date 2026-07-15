%% Manufactured-solution pressure test (verify BCs/source before transport).
%% -div(kappa grad p)=f, kappa=2x+1, f=-y^2-2x^2-x, p_exact=x*y^2/2.
%% BC: p=0 (x=0), p=y^2/2 (x=1), no-flow (y=0), flux -(2x^2+x) (y=1).
mrstRoot='D:\PhD\Research\Dissertation\mrst-2025a\SINTEF-AppliedCompSci-MRST-75749fa';
if exist('mrstModule','file')~=2, run(fullfile(mrstRoot,'startup.m')); end
mrstModule add incomp
gravity reset off

G=computeGeometry(cartGrid([128 128],[1 1]));
xc=G.cells.centroids;
rock.perm=2*xc(:,1)+1; rock.poro=ones(G.cells.num,1);
Tr=computeTrans(G,rock);

% distributed source: div(v)=f -> src rate = f*cellvolume  (f<0 = sink)
fval=-xc(:,2).^2-2*xc(:,1).^2-xc(:,1);
q_cell=fval.*G.cells.volumes;
src=addSource([],(1:G.cells.num)',q_cell,'sat',[1 0]);

% boundary faces
ff=boundaryFaces(G); fcx=G.faces.centroids(ff,1); fcy=G.faces.centroids(ff,2);
left=ff(abs(fcx)<1e-9); right=ff(abs(fcx-1)<1e-9);
top=ff(abs(fcy-1)<1e-9); bottom=ff(abs(fcy)<1e-9);

bc=addBC([],left,'pressure',zeros(numel(left),1));            % p=0
bc=addBC(bc,right,'pressure',G.faces.centroids(right,2).^2/2);% p=y^2/2
xt=G.faces.centroids(top,1);
topflux=(2*xt.^2+xt).*G.faces.areas(top);                     % try +into-domain; verify below
bc=addBC(bc,top,'flux',topflux);
% bottom: no-flow (default, no BC)

state=incompTPFA(initResSol(G,0),G,Tr,initSingleFluid('mu',1,'rho',1),'bc',bc,'src',src);
p=state.pressure; pex=xc(:,1).*xc(:,2).^2/2;
fprintf('\n=== pressure verification ===\n');
fprintf('p  L2 rel err = %.3e   Linf = %.3e\n', norm(p-pex)/max(norm(pex),eps), max(abs(p-pex)));

% boundary flux out of domain (outward normal)
N=G.faces.neighbors; isN2zero=N(ff,2)==0;
outb=state.flux(ff); outb(~isN2zero)=-outb(~isN2zero);   % >0 = OUT of domain
sL=sum(outb(ismember(ff,left))); sR=sum(outb(ismember(ff,right)));
sT=sum(outb(ismember(ff,top)));  sB=sum(outb(ismember(ff,bottom)));
fprintf('boundary net flux OUT (>0 out, <0 in):\n');
fprintf('  left x=0 = %+.5f (exact outflow +1/6=+0.1667)\n', sL);
fprintf('  right x=1= %+.5f (exact inflow -1/2=-0.5)\n', sR);
fprintf('  top  y=1 = %+.5f (exact inflow -7/6=-1.1667)\n', sT);
fprintf('  bottom   = %+.5f (exact 0)\n', sB);
Qin=-(sR+sT);   % total inflow magnitude (x=1 + y=1)
fprintf('total boundary inflow Q_in = %.5f (exact 5/3=1.6667)\n', Qin);
fprintf('PVI=0.3 -> T1 = %.5f ; suggest T2=0.20\n', 0.3*sum(poreVolume(G,rock))/Qin);
