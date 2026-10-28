"""Conditional predictive checks and a no-trial state-space alternative."""
from common import *
from scipy.optimize import minimize
from scipy.special import expit,logit
from statistics import cosinor
import json

def fitted_predictive_checks(bins):
    fits=pd.read_csv(OUT/'results/corrected_expected_model_fits.csv');pred=pd.read_csv(OUT/'results/corrected_expected_model_predictions.csv')
    best=fits.loc[fits.groupby('MouseID').pseudo_AIC_1000.idxmin()];rows=[];coverage=[]
    for _,r in best.iterrows():
        g=bins[bins.MouseID==r.MouseID].sort_values('bin_index');p=pred[(pred.MouseID==r.MouseID)&pred.model.eq(r.model)].sort_values('bin_index')
        rg=rng_for('predictive'+r.MouseID);n=p.N_bin.to_numpy(int);phat=p.predicted.to_numpy();y=rg.binomial(n,phat,size=(500,len(n)))
        y=np.divide(y,n,out=np.full(y.shape,np.nan),where=n>0);zt=g.ZT.to_numpy();obs=g.preference.to_numpy();ok=n>0
        X=np.column_stack([np.ones(ok.sum()),np.cos(zt[ok]*np.pi/12),np.sin(zt[ok]*np.pi/12)]);b=y[:,ok]@np.linalg.pinv(X).T
        amp=np.hypot(b[:,1],b[:,2]);phase=np.arctan2(b[:,2],b[:,1])*12/np.pi%24;cf=cosinor(zt,obs)
        simulated={'amplitude':amp,'SD':np.nanstd(y,axis=1,ddof=1),
                   'night_minus_day':np.nanmean(y[:,zt>=12],axis=1)-np.nanmean(y[:,zt<12],axis=1),
                   'adjacent_bin_change':np.nanmean(abs(np.diff(y,axis=1)),axis=1)}
        observed={'amplitude':cf['amplitude'],'SD':np.nanstd(obs,ddof=1),'night_minus_day':np.nanmean(obs[zt>=12])-np.nanmean(obs[zt<12]),'adjacent_bin_change':np.nanmean(abs(np.diff(obs)))}
        for metric,arr in simulated.items():
            lo,hi=np.nanquantile(arr,[.025,.975]);v=observed[metric];rows.append(dict(MouseID=r.MouseID,model=r.model,metric=metric,observed=v,predictive_mean=np.nanmean(arr),predictive_low=lo,predictive_high=hi,within_95=lo<=v<=hi))
        for i in range(len(g)):
            if n[i]==0 or not np.isfinite(obs[i]):continue
            lo,hi=np.nanquantile(y[:,i],[.025,.975]);coverage.append(dict(MouseID=r.MouseID,model=r.model,bin_index=i,observed=obs[i],predicted=phat[i],predictive_low=lo,predictive_high=hi,within_95=bool(lo<=obs[i]<=hi)))
        # Report phase error as a circular offset from the model's predicted peak.
        pfit=cosinor(zt,phat)
        if np.isfinite(pfit['phase_h']) and np.isfinite(cf['phase_h']):
            lo,hi=np.nanquantile(wrap_hours(phase-pfit['phase_h']),[.025,.975]);v=wrap_hours(cf['phase_h']-pfit['phase_h'])
            rows.append(dict(MouseID=r.MouseID,model=r.model,metric='phase_offset_h',observed=float(v),predictive_mean=0,predictive_low=lo,predictive_high=hi,within_95=bool(lo<=v<=hi)))
    rows=pd.DataFrame(rows);save_table(rows,'fitted_predictive_checks');save_table(pd.DataFrame(coverage),'fitted_predictive_bin_intervals')
    save_table(rows.groupby('metric').agg(n=('MouseID','size'),coverage_95=('within_95','mean')).reset_index(),'fitted_predictive_coverage')

def state_space_forecast(bins):
    rows=[];parameters=[]
    for mid,g in bins[bins.modeling_cohort].groupby('MouseID'):
        g=g.sort_values('Time_start');y=g.preference.to_numpy();total=g.total_g.to_numpy();t=g.ZT.to_numpy();elapsed=g.elapsed_h.to_numpy()
        train=np.arange(len(g))<6;valid=train&np.isfinite(y)&(total>0);test=~train&np.isfinite(y)&(total>0)
        if valid.sum()<4 or not test.any():continue
        z=logit(np.clip(y,1e-3,1-1e-3));weights=total/max(total[valid].mean(),1e-9)
        def kalman(par,forecast=False):
            m=par[0];P=1.;q,r=np.exp(par[1:]);loss=0.;predictions=[]
            for j in range(len(g)):
                P+=q
                if valid[j]:
                    R=r/max(weights[j],1e-6);S=P+R;err=z[j]-m;loss+=.5*(np.log(2*np.pi*S)+err*err/S);K=P/S;m+=K*err;P*=1-K
                predictions.append(expit(m/np.sqrt(1+np.pi*P/8)))
            return np.array(predictions) if forecast else loss
        initial=logit(np.clip(np.average(y[valid],weights=total[valid]),.001,.999))
        results=[minimize(kalman,[initial,np.log(q),np.log(.2)],method='L-BFGS-B',bounds=[(-8,8),(-12,2),(-8,3)],options={'maxiter':300}) for q in [.01,.3]]
        fit=min(results,key=lambda x:x.fun);pred=kalman(fit.x,True)
        parameters.append(dict(MouseID=mid,initial_logit=fit.x[0],process_variance=np.exp(fit.x[1]),observation_variance=np.exp(fit.x[2]),converged=bool(fit.success)))
        designs={'constant':np.ones((len(g),1)),'linear_drift':np.column_stack([np.ones(len(g)),(elapsed-12)/12]),
                 '24h_harmonic':np.column_stack([np.ones(len(g)),np.cos(t*np.pi/12),np.sin(t*np.pi/12)])}
        predictions={'local_level_state_space':pred}
        for name,X in designs.items():
            beta=np.linalg.lstsq(X[valid]*np.sqrt(total[valid])[:,None],y[valid]*np.sqrt(total[valid]),rcond=None)[0];predictions[name]=np.clip(X@beta,0,1)
        for model,p in predictions.items():
            rows.append(dict(MouseID=mid,model=model,n_train=int(valid.sum()),n_test=int(test.sum()),forward_mse=np.mean((y[test]-p[test])**2),
                forward_weighted_mse=np.average((y[test]-p[test])**2,weights=total[test])))
    save_table(pd.DataFrame(rows),'state_space_forward_validation');save_table(pd.DataFrame(parameters),'state_space_parameters')

