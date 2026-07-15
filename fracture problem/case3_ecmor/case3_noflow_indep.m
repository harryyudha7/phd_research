%% NO-FLOW variant: independent MRST explicitTransport -> overwrite sw_* in export.
mrstModule add hfm incomp ad-core
S=load('c:\Users\muchamad\mrst-project\case3_noflow_setup.mat');
G=S.G; T=S.T; fluid=S.fluid2; bc=S.bc2; nc=S.nc; sF=S.sF; T_final=S.T_final;
state=S.state2;
fprintf('loaded no-flow setup (no tip wells). T_final=%.5f\n', T_final);

nout=10; dTo=T_final/nout; tS=tic;
for kk=1:nout
    state=explicitTransport(state,G,dTo,G.rock,fluid,'bc',bc,'Trans',T);   % NO wells (no-flow tips)
    fprintf('  %2d/%d Sw_max=%.4f Sfrac_max=%.4f [%.1fs]\n', kk,nout, ...
            max(state.s(1:nc,1)), max(state.s(nc+1:end,1)), toc(tS));
end
s_matrix_e=state.s(1:nc,1); s_frac_e=state.s(nc+1:end,1);

%% overwrite sw_* (MRST explicitTransport) in the no-flow export; keep *_matched
E=load('c:\Users\muchamad\mrst-project\case3_mrst_export_noflow.mat');
rmse=@(a,b) sqrt(mean((a-b).^2));
fprintf('\n=== NO-FLOW: independent explicitTransport vs hand-code ===\n');
fprintf('matrix   Sw: indep max=%.4f  hand max=%.4f  RMSE=%.4e\n', max(s_matrix_e),max(E.sw_matrix_matched),rmse(s_matrix_e,E.sw_matrix_matched));
fprintf('fracture Sw: indep max=%.4f  hand max=%.4f  RMSE=%.4e\n', max(s_frac_e),max(E.sw_frac_matched),rmse(s_frac_e,E.sw_frac_matched));
E.sw_matrix=s_matrix_e(:); E.sw_frac=s_frac_e(:); E.s_matrix=s_matrix_e(:);
E.meta_transport_solver='MRST explicitTransport (SINTEF, independent) - explicit 1st-order upwind; no-flow fracture tips';
save('c:\Users\muchamad\mrst-project\case3_mrst_export_noflow.mat','-struct','E','-v7');
fprintf('updated case3_mrst_export_noflow.mat: sw_* := MRST explicitTransport (frac max %.4f)\n', max(E.sw_frac));
