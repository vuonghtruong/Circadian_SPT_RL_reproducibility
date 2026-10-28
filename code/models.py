"""Sensitivity models for interval data. Latent trials are assumptions, never observations.

The carry-over model retains the legacy expected-value update, fixes its state reset,
and shares the same predictor in fitting, simulations, and model checking.
"""
from common import *
from scipy.optimize import minimize
from scipy.special import expit,logit
from numba import njit
import json,time

MODEL_NAMES=['random','bias','linear_bias','step_bias','simple','simple_no_swap','reward','reward_no_swap','full','full_no_swap']
PARAM_NAMES={'random':[],'bias':['intercept'],'linear_bias':['intercept','slope'],'step_bias':['intercept','night_effect'],
    'simple':['log10_alpha','log_beta','base_reward'],'reward':['log10_alpha','log_beta','base_reward','reward_amplitude','phase'],
    'full':['log10_alpha','log_beta','base_reward','reward_amplitude','phase','temperature_amplitude']}
for m in ['simple','reward','full']:PARAM_NAMES[m+'_no_swap']=PARAM_NAMES[m]
def bounds_for(model):
    if model=='random':return []
    if model=='bias':return [(-8,8)]
    if model in ['linear_bias','step_bias']:return [(-8,8),(-8,8)]
    b=[(-4,0),(np.log(.1),np.log(20)),(.05,.95)]
    if model.startswith(('reward','full')):b += [(0,.45),(0,2*np.pi)]
    if model.startswith('full'):b += [(0,.95)]
    return b

def allocate(total,N,schedule='observed',rng=None):
    w=np.asarray(total,float)
    if schedule=='uniform':w=np.ones(len(w))
    if schedule=='shuffled':w=(rng or rng_for('schedule')).permutation(w)
    p=w/w.sum();ideal=N*p;n=np.floor(ideal).astype(int)
    order=np.argsort(-(ideal-n),kind='stable');n[order[:N-n.sum()]]+=1
    assert n.sum()==N and np.all(n>=0)
    return n

@njit(cache=True)
def expected_rl_core(par,zt,bin_index,n,switch=True,reset=False,shape=0,record=False):
    alpha=10.**par[0];beta0=np.exp(par[1]);base=par[2]
    amp=par[3] if len(par)>3 else 0.;phase=par[4] if len(par)>4 else 0.;ampb=par[5] if len(par)>5 else 0.
    v0=0.;v1=0.;out=np.zeros(len(zt));trace=np.zeros(n.sum() if record else 0);cursor=0
    for i in range(len(zt)):
        if reset:v0=0.;v1=0.
        c=np.sin(zt[i]*np.pi/12+phase)
        if shape==1:c=1. if c>=0 else -1.
        beta=max(.2,beta0*(1-ampb*c));r=min(.95,max(.05,base+amp*c))
        sucrose_left=(bin_index[i]%2==0) if switch else True
        r0=r if sucrose_left else 1-r;r1=1-r0;psum=0.
        for _ in range(n[i]):
            x=min(700.,max(-700.,beta*(v0-v1)));p0=.01+.98/(1+np.exp(-x));p1=1-p0
            preference=p0 if sucrose_left else p1
            psum+=preference
            if record:trace[cursor]=preference;cursor+=1
            # Legacy expected-update rule; this is not the sampled-action RW rule.
            v0+=alpha*(p0*r0-v0);v1+=alpha*(p1*r1-v1)
        if n[i]>0:out[i]=psum/n[i]
        else:
            x=min(700.,max(-700.,beta*(v0-v1)));p0=.01+.98/(1+np.exp(-x));out[i]=p0 if sucrose_left else 1-p0
    return out,trace

@njit(cache=True)
def expected_rl(par,zt,bin_index,n,switch=True,reset=False,shape=0):
    return expected_rl_core(par,zt,bin_index,n,switch,reset,shape,False)[0]

def predict(model,par,zt,elapsed,idx,n,shape=0,reset=False):
    if model=='random':return np.full(len(zt),.5)
    if model=='bias':return np.full(len(zt),expit(par[0]))
    if model=='linear_bias':return expit(par[0]+par[1]*(elapsed-12)/12)
    if model=='step_bias':return expit(par[0]+par[1]*(zt>=12))
    return expected_rl(np.asarray(par),zt,idx,n,not model.endswith('no_swap'),reset,shape)

def fit_one(model,zt,elapsed,idx,n,y,weights,label,starts=6,shape=0,warm_seeds=None):
    valid=np.isfinite(y)&(weights>0);w=weights[valid]/weights[valid].sum();yy=y[valid]
    def objective(par):
        p=np.clip(predict(model,par,zt,elapsed,idx,n,shape)[valid],1e-10,1-1e-10)
        return -np.sum(w*(yy*np.log(p)+(1-yy)*np.log1p(-p)))
    if model in ['random','bias']:
        par=np.array([]) if model=='random' else np.array([np.clip(logit(np.clip(np.average(yy,weights=w),1e-9,1-1e-9)),-8,8)])
        return par,objective(par),dict(converged=True,nit=0,nfev=1,starts=1,improvement=0.,near_best_starts=1,at_boundary=False)
    bounds=np.array(bounds_for(model));rg=rng_for(label+model)
    seeds=rg.uniform(bounds[:,0],bounds[:,1],(starts,len(bounds)))
    pref=np.clip(np.average(yy,weights=w),.06,.94)
    if 'bias' in model:seeds[0]=[logit(pref),0]
    else:
        for j in range(min(3,starts)):
            seeds[j,:3]=[-3+j,np.log(4),pref]
            if len(bounds)>3:seeds[j,3:5]=[.15,(j+1)*np.pi/2]
            if len(bounds)>5:seeds[j,5]=.25
    # A richer nested model must attain at least the embedded parent fit.
    # Random starts alone previously missed this feasible solution for some Full fits.
    if model.startswith(('reward','full')):
        if warm_seeds is None:
            parent=('simple' if model.startswith('reward') else 'reward')+('_no_swap' if model.endswith('no_swap') else '')
            parent_par,_,_=fit_one(parent,zt,elapsed,idx,n,y,weights,label,starts=starts,shape=shape)
            warm_seeds=[np.r_[parent_par,0.,np.pi] if model.startswith('reward') else np.r_[parent_par,0.]]
        seeds=np.vstack([seeds,np.asarray(warm_seeds)])
    results=[];initial=[]
    for seed in seeds:
        initial.append(objective(seed))
        result=minimize(objective,seed,method='L-BFGS-B',bounds=bounds,options={'maxiter':500,'ftol':1e-11,'gtol':1e-6,'maxls':40})
        if result.fun>initial[-1]+1e-12:
            result.x=seed.copy();result.fun=initial[-1];result.success=False
        results.append(result)
    best=min(results,key=lambda r:r.fun);boundary=np.any(np.minimum(best.x-bounds[:,0],bounds[:,1]-best.x)<1e-5)
    return best.x,best.fun,dict(converged=bool(best.success),nit=int(best.nit),nfev=int(best.nfev),starts=len(seeds),
        improvement=float(min(initial)-best.fun),near_best_starts=sum(r.fun-best.fun<1e-5 for r in results),at_boundary=bool(boundary))

def model_refits(bins):
    fitrows=[];predrows=[];t0=time.time();pal=bins[bins.modeling_cohort]
    for j,(mid,g) in enumerate(pal.groupby('MouseID')):
        g=g.sort_values('Time_start');zt=g.ZT.to_numpy(float);elapsed=g.elapsed_h.to_numpy(float);idx=g.bin_index.to_numpy(np.int64)
        y=g.preference.to_numpy(float);weights=g.total_g.to_numpy(float);n=allocate(weights,1000)
        for model in MODEL_NAMES:
            par,loss,diag=fit_one(model,zt,elapsed,idx,n,y,weights,mid)
            fitrows.append(dict(MouseID=mid,model=model,k=len(par),N_assumed=1000,cross_entropy=loss,
                pseudo_AIC_100=200*loss+2*len(par),pseudo_AIC_1000=2000*loss+2*len(par),pseudo_AIC_10000=20000*loss+2*len(par),
                parameters=json.dumps(par.tolist()),**diag))
            pred=predict(model,par,zt,elapsed,idx,n)
            predrows.extend(dict(MouseID=mid,model=model,bin_index=int(idx[i]),ZT=zt[i],observed=y[i],predicted=pred[i],N_bin=int(n[i])) for i in range(len(g)))
        if (j+1)%25==0:print(f'Refit {j+1}/{pal.MouseID.nunique()} sessions ({time.time()-t0:.0f}s)',flush=True)
    fits=pd.DataFrame(fitrows);pred=pd.DataFrame(predrows);save_table(fits,'corrected_expected_model_fits');save_table(pred,'corrected_expected_model_predictions')
    counts=[]
    for scale in [100,1000,10000]:
        best=fits.loc[fits.groupby('MouseID')[f'pseudo_AIC_{scale}'].idxmin()]
        counts.extend(dict(pseudo_count=scale,model=m,n=int(n)) for m,n in best.model.value_counts().items())
    save_table(pd.DataFrame(counts),'pseudo_count_penalty_sensitivity')
    # Held-out contiguous three-hour pairs: four folds. Trial availability remains
    # conditioned on observed total intake; held-out preference never enters fitting.
    cv=[]
    for j,(mid,g) in enumerate(pal.groupby('MouseID')):
        g=g.sort_values('Time_start');zt=g.ZT.to_numpy(float);elapsed=g.elapsed_h.to_numpy(float);idx=g.bin_index.to_numpy(np.int64)
        y=g.preference.to_numpy(float);total=g.total_g.to_numpy(float);n=allocate(total,1000)
        for model in ['bias','linear_bias','step_bias','simple','reward','full']:
            for fold in range(4):
                held=idx//2==fold;weights=total.copy();weights[held]=0
                par,loss,diag=fit_one(model,zt,elapsed,idx,n,y,weights,mid+str(fold),starts=3)
                p=predict(model,par,zt,elapsed,idx,n);ok=held&np.isfinite(y)&(total>0)
                if not ok.any():continue
                cv.append(dict(MouseID=mid,model=model,fold=fold,heldout_weighted_mse=float(np.average((y[ok]-p[ok])**2,weights=total[ok])),
                    heldout_mse=float(np.mean((y[ok]-p[ok])**2)),converged=diag['converged']))
        if (j+1)%50==0:print(f'Blocked CV {j+1}/{pal.MouseID.nunique()}',flush=True)
    save_table(pd.DataFrame(cv),'expected_model_blocked_cv')
    return fits,pred

def null_controls(bins,repeats=500):
    from statistics import cosinor
    pal=bins[bins.modeling_cohort];rows=[];aggregate=[];example=[]
    for N in [100,1000,10000]:
        for schedule in ['observed','uniform','shuffled']:
            simulations=[];daydiff=[]
            for mid,g in pal.groupby('MouseID'):
                g=g.sort_values('ZT');rg=rng_for(mid+schedule+str(N));total=g.total_g.to_numpy(float);p=g.reward_g.sum()/g.total_g.sum()
                counts=allocate(total,N,schedule,rg)
                # Constant underlying preference; only precision changes with bin count.
                Y=rg.binomial(counts,p,size=(repeats,len(counts))).astype(float)
                Y=np.divide(Y,counts,out=np.full_like(Y,np.nan),where=counts>0)
                t=g.ZT.to_numpy(float);valid=counts>0
                X=np.column_stack([np.ones(valid.sum()),np.cos(t[valid]*np.pi/12),np.sin(t[valid]*np.pi/12)])
                coef=Y[:,valid]@np.linalg.pinv(X).T;amp=np.hypot(coef[:,1],coef[:,2])
                obs=cosinor(t,g.preference)['amplitude']
                # Pearson correlations are descriptive under this heteroscedastic null.
                centered=Y-np.nanmean(Y,axis=1,keepdims=True);a=total-total.mean()
                denominator=np.sqrt(np.sum(a*a)*np.nansum(centered**2,axis=1))
                r=np.divide(np.nansum(centered*a,axis=1),denominator,out=np.full(repeats,np.nan),where=denominator>1e-12)
                rows.append(dict(MouseID=mid,N_assumed=N,schedule=schedule,observed_amplitude=obs,null_amplitude_median=np.nanmedian(amp),
                    null_amplitude_95=np.nanquantile(amp,.95),p_amplitude=(1+np.sum(amp>=obs))/(repeats+1),null_r_median=np.nanmedian(r)))
                simulations.append(Y);daydiff.append(np.nanmean(Y[:,t>=12],axis=1)-np.nanmean(Y[:,t<12],axis=1))
            # All CSV sessions use the same eight ZT bin centers. Mean ratio per session.
            population=np.nanmean(np.stack(simulations),axis=0);day=np.mean(np.stack(daydiff),axis=0)
            X=np.column_stack([np.ones(8),np.cos(t*np.pi/12),np.sin(t*np.pi/12)]);co=population@np.linalg.pinv(X).T
            amp=np.hypot(co[:,1],co[:,2])
            observed=pal.groupby('ZT').preference.mean();obs=cosinor(observed.index,observed.values)['amplitude']
            aggregate.append(dict(N_assumed=N,schedule=schedule,observed_group_amplitude=obs,null_amp_median=np.median(amp),null_amp_95=np.quantile(amp,.95),
                p_group_amplitude=(1+np.sum(amp>=obs))/(repeats+1),night_minus_day_median=np.median(day),night_minus_day_ci_low=np.quantile(day,.025),night_minus_day_ci_high=np.quantile(day,.975)))
            for i,tt in enumerate(t):example.append(dict(N_assumed=N,schedule=schedule,ZT=tt,mean=np.nanmean(population[:,i]),low=np.nanquantile(population[:,i],.025),high=np.nanquantile(population[:,i],.975)))
    null=pd.DataFrame(rows);null['q_amplitude']=null.groupby(['N_assumed','schedule']).p_amplitude.transform(bh)
    save_table(null,'constant_bias_null_by_session');save_table(pd.DataFrame(aggregate),'constant_bias_null_population');save_table(pd.DataFrame(example),'constant_bias_null_profiles')

def illustrative_simulations():
    rows=[];traces=[];controls=[];event_rows=[]
    zt=np.arange(1.5,24,3);idx=np.arange(8,dtype=np.int64);elapsed=zt.copy()
    pars={'random':[],'bias':[.7],'simple':[-np.log10(20),np.log(4),.65],
          'reward':[-np.log10(20),np.log(4),.65,.25,np.pi],'full':[-np.log10(20),np.log(4),.65,.25,np.pi,.5]}
    for model,par in pars.items():
        for schedule,counts in [('observed',np.r_[np.full(4,50),np.full(4,200)]),('uniform',np.full(8,125)),('shuffled',np.array([200,50,200,50,50,200,50,200]))]:
            p=predict(model,np.array(par),zt,elapsed,idx,counts);rg=rng_for(model+schedule)
            event_p=np.repeat(p,counts) if model in ['random','bias'] else expected_rl_core(np.array(par),zt,idx,counts,True,False,0,True)[1]
            simulated_events=rg.binomial(1,event_p,size=(500,len(event_p)))
            y=np.column_stack([part.mean(axis=1) for part in np.split(simulated_events,np.cumsum(counts)[:-1],axis=1)])
            controls.append(dict(model=model,schedule=schedule,expected_day=p[:4].mean(),expected_night=p[4:].mean(),expected_night_minus_day=p[4:].mean()-p[:4].mean(),
                simulated_day=y[:,:4].mean(),simulated_night=y[:,4:].mean()))
            if schedule=='observed':
                times=np.concatenate([i*3+(np.arange(nn)+.5)*3/nn for i,nn in enumerate(counts)])
                event_p=np.repeat(p,counts) if model in ['random','bias'] else expected_rl_core(np.array(par),zt,idx,counts,True,False,0,True)[1]
                # The dense trace exposes the same expected-update recurrence used in fitting.
                reconstructed=np.array([part.mean() for part in np.split(event_p,np.cumsum(counts)[:-1])])
                assert np.allclose(reconstructed,p,atol=1e-12)
                event_choice=rng_for('trace'+model).binomial(1,event_p)
                smooth=pd.Series(event_choice).rolling(31,center=True,min_periods=10).mean().to_numpy()
                event_rows.extend(dict(model=model,event=j,time_h=float(t),expected=float(event_p[j]),choice=int(event_choice[j]),rolling_choice=float(smooth[j])) for j,t in enumerate(times))
                for i,t in enumerate(zt):rows.append(dict(model=model,ZT=t,expected=p[i],mean=y[:,i].mean(),sem=stats.sem(y[:,i]),n=int(counts[i])))
                # Bin-level predictor is the fitting model; retain explicit discontinuities.
                for run in range(50):
                    for phase,slice_ in [('Day',slice(0,4)),('Night',slice(4,8))]:
                        traces.append(dict(model=model,run=run,phase=phase,preference=y[run,slice_].mean()))
    save_table(pd.DataFrame(rows),'illustrative_profiles');save_table(pd.DataFrame(traces),'illustrative_day_night');save_table(pd.DataFrame(controls),'trial_availability_counterfactual');save_table(pd.DataFrame(event_rows),'illustrative_event_trajectories')

def recovery_and_sensitivity(bins,fits):
    """Small explicitly scoped simulation recovery, including all five model classes.

    Recovery cannot validate unobserved drinking bouts; it tests this mathematical model.
    """
    rows=[];sensitivity=[]
    eligible=bins[bins.modeling_cohort].groupby('MouseID').filter(lambda g:g.total_g.gt(0).all())
    ids=sorted(eligible.MouseID.unique());sample=[ids[i] for i in np.linspace(0,len(ids)-1,12,dtype=int)]
    for mid in sample:
        g=eligible[eligible.MouseID==mid].sort_values('Time_start');zt=g.ZT.to_numpy(float);elapsed=g.elapsed_h.to_numpy(float);idx=g.bin_index.to_numpy(np.int64);y=g.preference.to_numpy(float);w=g.total_g.to_numpy(float)
        for N in [100,1000,10000]:
            for shape in [0,1]:
                for model in ['bias','linear_bias','simple','reward','full']:
                    par,loss,diag=fit_one(model,zt,elapsed,idx,allocate(w,N),y,w,mid+str(N)+str(shape),starts=4,shape=shape)
                    sensitivity.append(dict(MouseID=mid,N_assumed=N,shape='sine' if shape==0 else 'square',model=model,cross_entropy=loss,pseudo_AIC=2*N*loss+2*len(par),converged=diag['converged']))
    save_table(pd.DataFrame(sensitivity),'latent_count_shape_refit_sensitivity')
    zt=np.arange(1.5,24,3);idx=np.arange(8,dtype=np.int64);counts=np.r_[np.full(4,50),np.full(4,200)]
    truths={'bias':[.7],'linear_bias':[.5,.8],'simple':[-1.5,np.log(4),.7],
            'reward':[-1.5,np.log(4),.7,.2,2.],'full':[-1.5,np.log(4),.7,.2,2.,.5]}
    parameter_rows=[]
    for true_model,true_par in truths.items():
        p=predict(true_model,np.array(true_par),zt,zt,idx,counts);rg=rng_for('recovery'+true_model)
        for rep in range(20):
            y=rg.binomial(counts,p)/counts
            candidates=[]
            for model in truths:
                par,loss,diag=fit_one(model,zt,zt,idx,counts,y,counts,true_model+str(rep),starts=4)
                candidates.append((2*1000*loss+2*len(par),model))
                if model==true_model:
                    for pname,truth,estimate in zip(PARAM_NAMES[model],true_par,par):parameter_rows.append(dict(model=model,replicate=rep,parameter=pname,true_value=truth,estimate=estimate))
            best=min(candidates)[1];rows.append(dict(true_model=true_model,replicate=rep,selected_model=best))
    save_table(pd.DataFrame(rows),'model_recovery');save_table(pd.DataFrame(parameter_rows),'parameter_recovery')

