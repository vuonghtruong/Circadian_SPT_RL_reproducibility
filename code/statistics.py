"""Session-level inference, clearly separated from historical figure statistics."""
from common import *
import json, warnings
from scipy.special import expit

def cosinor(t,y):
    t,y=np.asarray(t,float),np.asarray(y,float); good=np.isfinite(t)&np.isfinite(y);t,y=t[good],y[good]
    out=dict(n_bins=len(y),mesor=np.nan,amplitude=np.nan,phase_h=np.nan,p=np.nan,r2=np.nan,beta_cos=np.nan,beta_sin=np.nan,
             amplitude_low=np.nan,amplitude_high=np.nan,phase_ci_halfwidth_h=12.,phase_identified=False)
    if len(y)<6:return out
    X=np.column_stack([np.ones(len(t)),np.cos(t*np.pi/12),np.sin(t*np.pi/12)])
    b=np.linalg.lstsq(X,y,rcond=None)[0]; err=y-X@b; rss=err@err; tss=np.sum((y-y.mean())**2)
    amp=np.hypot(b[1],b[2]); phase=np.arctan2(b[2],b[1])*12/np.pi%24
    if tss<1e-16:p=1.;r2=0.;phase=np.nan
    else:
        F=max((tss-rss)/2,0)/max(rss/(len(y)-3),1e-30);p=stats.f.sf(F,2,len(y)-3);r2=1-rss/tss
    covariance=rss/(len(y)-3)*np.linalg.inv(X.T@X)
    amp_grad=np.array([b[1],b[2]])/amp if amp>1e-12 else np.zeros(2)
    amp_se=np.sqrt(max(amp_grad@covariance[1:,1:]@amp_grad,0))
    critical=stats.t.ppf(.975,len(y)-3)
    phasegrad=np.array([-b[2],b[1]])/max(amp**2,1e-24)
    phase_se=np.sqrt(max(phasegrad@covariance[1:,1:]@phasegrad,0))*12/np.pi
    identified=bool(p<.05 and amp>1e-10)
    out.update(mesor=b[0],amplitude=amp,phase_h=phase,p=p,r2=r2,beta_cos=b[1],beta_sin=b[2],
        amplitude_low=max(0,amp-critical*amp_se) if identified else 0.,amplitude_high=amp+critical*amp_se,
        phase_ci_halfwidth_h=min(12.,critical*phase_se) if identified else 12.,phase_identified=identified)
    return out

def group_summary(g,label,n_boot=2000,n_perm=1999):
    rg=rng_for(label); subjects=[d for _,d in g.groupby('MouseID')]
    fits=[cosinor(d.ZT,d.preference) for d in subjects]
    eligible=[(d,f) for d,f in zip(subjects,fits) if f['n_bins']>=6 and np.isfinite(f['phase_h'])]
    if not eligible:return None
    coeff=np.array([[f['beta_cos'],f['beta_sin']] for _,f in eligible]);n=len(coeff)
    avg=coeff.mean(0);amp=np.hypot(*avg);phase=np.arctan2(avg[1],avg[0])*12/np.pi%24
    if n>2 and np.linalg.matrix_rank(np.cov(coeff,rowvar=False))==2:
        T2=n*avg@np.linalg.inv(np.cov(coeff,rowvar=False))@avg
        p=stats.f.sf((n-2)/(2*(n-1))*T2,2,n-2)
    else:p=np.nan
    ids=rg.integers(0,n,(n_boot,n)); b=coeff[ids].mean(1)
    boot_amp=np.hypot(b[:,0],b[:,1]);boot_phase=np.arctan2(b[:,1],b[:,0])*12/np.pi%24
    offset=wrap_hours(boot_phase-phase);lo,hi=np.quantile(offset,[.025,.975])
    phases=np.array([f['phase_h'] for _,f in eligible]);R=resultant(phases)
    null_complex=np.zeros((n_perm,n),complex)
    for j,(d,f) in enumerate(eligible):
        t=d.ZT.to_numpy();y=d.preference.to_numpy();valid=np.isfinite(y);t,y=t[valid],y[valid]
        X=np.column_stack([np.ones(len(t)),np.cos(t*np.pi/12),np.sin(t*np.pi/12)])
        Y=rg.permuted(np.tile(y,(n_perm,1)),axis=1)
        perm_coeff=Y@np.linalg.pinv(X).T
        null_complex[:,j]=np.exp(1j*np.arctan2(perm_coeff[:,2],perm_coeff[:,1]))
    null_vector=null_complex.mean(1);boot_vector=np.exp(2j*np.pi*phases[ids]/24).mean(1)
    null_R=abs(null_vector);boot_R=abs(boot_vector)
    resamples=pd.concat([pd.DataFrame(dict(group=label,method=method,replicate=np.arange(len(v)),phase_h=np.angle(v)*12/np.pi%24,R=abs(v))) for method,v in [('bootstrap',boot_vector),('permutation',null_vector)]],ignore_index=True)
    save_table(resamples,'acrophase_resamples_'+label.replace(' ','_'))
    means=g.groupby('ZT').preference.agg(['mean','sem']).reset_index(); legacy=cosinor(means.ZT,means['mean'])
    # WLS F-test for comparison with the manuscript's stated method.
    X=np.column_stack([np.ones(len(means)),np.cos(means.ZT*np.pi/12),np.sin(means.ZT*np.pi/12)])
    w=1/np.maximum(means['sem'].fillna(0).to_numpy()**2,1e-8);y=means['mean'].to_numpy()
    beta=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
    rss=np.sum(w*(y-X@beta)**2);rss0=np.sum(w*(y-np.average(y,weights=w))**2)
    wls_p=stats.f.sf(max(0,(rss0-rss)/2)/max(rss/(len(y)-3),1e-30),2,len(y)-3)
    return dict(group=label,n_sessions=g.MouseID.nunique(),n_phase_eligible=n,mesor=np.mean([f['mesor'] for _,f in eligible]),
        amplitude=amp,amplitude_low=np.quantile(boot_amp,.025) if p<.05 else 0.,amplitude_high=np.quantile(boot_amp,.975),phase_h=phase,
        phase_ci_low_offset_h=lo if p<.05 else -12.,phase_ci_high_offset_h=hi if p<.05 else 12.,phase_uncertain=not(p<.05),p_population=p,
        p_group_mean_OLS=legacy['p'],p_group_mean_WLS=wls_p,group_mean_amplitude=legacy['amplitude'],group_mean_mesor=legacy['mesor'],
        group_mean_phase_h=legacy['phase_h'],R=R,emergence=R/null_R.mean(),p_phase_permutation=(1+np.sum(null_R>=R))/(n_perm+1),
        dominance_overlap=float(np.mean(boot_R[:,None]<=null_R[None,:])),legacy_n_sig=sum(f['p']<.05 for _,f in eligible))

def permutation_difference(x,y,label,n_perm=9999):
    x,y=np.asarray(x),np.asarray(y);rg=rng_for(label)
    if min(len(x),len(y))<2:return np.nan
    def studentized(a,b):
        se=np.sqrt(np.var(a,ddof=1)/len(a)+np.var(b,ddof=1)/len(b))
        return (a.mean()-b.mean())/max(se,1e-15)
    obs=studentized(x,y);pooled=np.r_[x,y]; count=0
    for _ in range(n_perm):
        p=rg.permutation(pooled);count+=abs(studentized(p[:len(x)],p[len(x):]))>=abs(obs)-1e-14
    return (count+1)/(n_perm+1)

def compute_statistics(bins):
    bins=bins[~bins.excluded_AD].copy()
    rows=[]
    for mid,g in bins.groupby('MouseID'):
        for metric in METRICS:
            rows.append(dict(MouseID=mid,metric=metric,**cosinor(g.ZT,g[metric])))
    individual=pd.DataFrame(rows)
    individual['q']=individual.groupby('metric').p.transform(bh)
    meta=bins[['MouseID','Sex','Age','age_group','Genotype','Condition','Condition_real','fluid','legacy_age_cohort','palatable_cohort','modeling_cohort','inferred_animal_id']].drop_duplicates()
    individual=individual.merge(meta,on='MouseID',validate='many_to_one');save_table(individual,'individual_cosinor')
    cohorts={f'{s} {a}':g for (s,a),g in bins[bins.legacy_age_cohort].groupby(['Sex','age_group'])}
    gr=[group_summary(g,k) for k,g in cohorts.items()]; group=pd.DataFrame([x for x in gr if x])
    for c in ['p_population','p_group_mean_OLS','p_group_mean_WLS','p_phase_permutation']:group[c.replace('p_','q_',1)]=bh(group[c])
    save_table(group,'group_cosinor')
    # Additional group profiles keep individual condition/genotype identities visible.
    condition=[]
    for key,g in bins.groupby(['Condition','Genotype']):
        for metric in METRICS:
            m=g.groupby('ZT')[metric].mean().reset_index(); condition.append(dict(Condition=key[0],Genotype=key[1],metric=metric,n_sessions=g.MouseID.nunique(),**cosinor(m.ZT,m[metric])))
    condition=pd.DataFrame(condition);condition['q']=bh(condition.p);save_table(condition,'condition_cosinor')
    # Paired day-night contrasts use session as the independent unit.
    dn=[];paired=[]
    for cohort,g in [('age_figures',bins[bins.legacy_age_cohort]),('all_palatable',bins[bins.palatable_cohort]),('modeling',bins[bins.modeling_cohort])]:
        for metric in METRICS:
            x=g.groupby(['MouseID','phase'])[metric].mean().unstack().dropna();d=x.Night-x.Day;n=len(d)
            t,p=stats.ttest_rel(x.Day,x.Night);crit=stats.t.ppf(.975,n-1)
            dn.append(dict(cohort=cohort,metric=metric,n=n,day_mean=x.Day.mean(),night_mean=x.Night.mean(),night_minus_day=d.mean(),
                ci_low=d.mean()-crit*stats.sem(d),ci_high=d.mean()+crit*stats.sem(d),t_day_minus_night=t,p=p,dz=d.mean()/d.std(ddof=1),
                wilcoxon_p=stats.wilcoxon(d).pvalue if np.any(d) else 1.))
            paired.append(x.reset_index().assign(cohort=cohort,metric=metric))
    dn=pd.DataFrame(dn);dn['q']=dn.groupby('cohort').p.transform(bh);save_table(dn,'day_night_tests');save_table(pd.concat(paired),'day_night_by_session')
    # Ratio of sums is reported separately from the arithmetic mean of ratios.
    sums=bins.groupby(['MouseID','phase'])[['reward_g','total_g']].sum();sums['ratio_of_sums']=sums.reward_g/sums.total_g.replace(0,np.nan)
    save_table(sums.reset_index(),'phase_ratio_of_sums')
    corr=[]
    for mid,g in bins[bins.modeling_cohort].groupby('MouseID'):
        g=g.dropna(subset=['preference']);n=len(g)
        if n>=3 and g.total_g.std()>1e-12 and g.preference.std()>1e-12:
            result=stats.pearsonr(g.total_g,g.preference);ci=result.confidence_interval()
            corr.append(dict(MouseID=mid,n_bins=n,r=result.statistic,p=result.pvalue,ci_low=ci.low,ci_high=ci.high))
        else:corr.append(dict(MouseID=mid,n_bins=n,r=np.nan,p=np.nan,ci_low=np.nan,ci_high=np.nan))
    corr=pd.DataFrame(corr);corr['q']=bh(corr.p);save_table(corr,'intake_preference_correlations')
    # Paired circular lag: sign flips of sin(delta), not a linear t-test on clock hours.
    wide=individual.pivot(index='MouseID',columns='metric',values=['phase_h','p','q']).reset_index()
    phase=pd.DataFrame(dict(MouseID=wide['MouseID'],total_phase=wide[('phase_h','total_g')],pref_phase=wide[('phase_h','preference')],
        total_p=wide[('p','total_g')],pref_p=wide[('p','preference')],total_q=wide[('q','total_g')],pref_q=wide[('q','preference')]))
    phase['lag_h']=wrap_hours(phase.pref_phase-phase.total_phase);phase=phase.merge(meta,on='MouseID',validate='one_to_one')
    save_table(phase,'paired_phases')
    lagrows=[]
    for scope,g0 in [('all_palatable',phase[phase.palatable_cohort]),('modeling',phase[phase.modeling_cohort]),*[(f'{s} {a}',g) for (s,a),g in phase[phase.legacy_age_cohort].groupby(['Sex','age_group'])],*[(str(k),g) for k,g in phase[phase.palatable_cohort].groupby('Condition')]]:
        for subset,g in [('all_estimable',g0),('both_nominal_rhythmic',g0[(g0.total_p<.05)&(g0.pref_p<.05)]),('both_FDR_rhythmic',g0[(g0.total_q<.05)&(g0.pref_q<.05)])]:
            g=g.dropna(subset=['lag_h']);n=len(g)
            if n<3:
                lagrows.append(dict(group=scope,subset=subset,n=n,lag_h=np.nan,p=np.nan,ci_low_offset_h=np.nan,ci_high_offset_h=np.nan,locking_R=np.nan,p_pairing=np.nan));continue
            rg=rng_for(scope+subset);theta=g.lag_h.to_numpy()*np.pi/12;mu=np.angle(np.mean(np.exp(1j*theta)))*12/np.pi
            s=np.sin(theta);obs=abs(s.mean())
            if n<=12:
                signs=1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1);p=np.mean(abs((signs*s).mean(1))>=obs-1e-14)
            else:
                signs=rg.choice([-1,1],size=(9999,n));p=(1+np.sum(abs((signs*s).mean(1))>=obs-1e-14))/10000
            boot=theta[rg.integers(0,n,(2000,n))];bm=np.angle(np.mean(np.exp(1j*boot),axis=1))*12/np.pi
            low,high=np.quantile(wrap_hours(bm-mu),[.025,.975]);R=resultant(g.lag_h)
            a=g.total_phase.to_numpy();b=g.pref_phase.to_numpy();rp=[resultant(rg.permutation(b)-a) for _ in range(1999)]
            lagrows.append(dict(group=scope,subset=subset,n=n,lag_h=mu,p=p,ci_low_offset_h=low,ci_high_offset_h=high,locking_R=R,p_pairing=(1+sum(x>=R for x in rp))/2000))
    lag=pd.DataFrame(lagrows);lag['q']=lag.groupby('subset').p.transform(bh);lag['q_pairing']=lag.groupby('subset').p_pairing.transform(bh);save_table(lag,'paired_circular_tests')
    fluid_analysis(bins)
    sensitivity_analysis(bins)
    print('Statistics complete',flush=True)
    return individual,group


def fluid_analysis(bins):
    import statsmodels.formula.api as smf
    data=bins.dropna(subset=['preference','Sex','Age']).copy()
    # Restrict to standard environmental condition; genotype retained for sensitivity.
    data=data[data.Condition_real=='Control'];data['fluid']=pd.Categorical(data.fluid,categories=['water','glucose','sucrose','saccharin'])
    rows=[]; logs=[]
    for scope,df in [('all_genotypes',data),('Control_genotype',data[data.Genotype=='Control'])]:
        with warnings.catch_warnings(record=True) as caught:
            try:
                fit=smf.mixedlm('preference ~ C(fluid)*C(Sex)',df,groups=df.inferred_animal_id).fit(reml=False,method='powell',maxiter=500,disp=False)
                logs.append(scope+'\n'+str(fit.summary())+'\n'+'\n'.join(str(x.message) for x in caught))
                for term in fit.fe_params.index:
                    rows.append(dict(scope=scope,model='bin_level_animal_random_intercept',term=term,estimate=fit.params[term],se=fit.bse[term],p=fit.pvalues[term],
                        ci_low=fit.conf_int().loc[term,0],ci_high=fit.conf_int().loc[term,1],n_sessions=df.MouseID.nunique(),n_animals=df.inferred_animal_id.nunique(),n_bins=len(df),converged=fit.converged))
            except Exception as e:logs.append(scope+' failed: '+repr(e))
        session=df.groupby(['MouseID','inferred_animal_id','fluid','Sex','Age'],observed=True).preference.mean().reset_index()
        fit=smf.ols('preference ~ C(fluid)*C(Sex) + Age',session).fit(cov_type='cluster',cov_kwds={'groups':session.inferred_animal_id},use_t=True)
        for term in fit.params.index:
            rows.append(dict(scope=scope,model='session_OLS_animal_cluster_age_adjusted',term=term,estimate=fit.params[term],se=fit.bse[term],p=fit.pvalues[term],
                ci_low=fit.conf_int().loc[term,0],ci_high=fit.conf_int().loc[term,1],n_sessions=len(session),n_animals=session.inferred_animal_id.nunique(),n_bins=len(session),converged=True))
    result=pd.DataFrame(rows);result['q']=result.groupby(['scope','model']).p.transform(bh);save_table(result,'fluid_model_coefficients')
    (OUT/'audit/fluid_model_diagnostics.txt').write_text('\n\n'.join(logs),encoding='utf-8')
    # Reproduce nearest-age, without-replacement matching; record pair differences.
    session=data.groupby(['MouseID','fluid','Sex','Age'],observed=True).total_g.sum().reset_index()
    pairs=[]
    for sex,g in session.groupby('Sex'):
        water=g[g.fluid=='water'].sort_values('MouseID');pal=g[g.fluid.isin(['sucrose','saccharin','glucose'])].copy()
        for _,r in water.iterrows():
            if pal.empty:break
            idx=(pal.Age-r.Age).abs().idxmin();p=pal.loc[idx];pal=pal.drop(idx)
            pairs.append(dict(Sex=sex,water_session=r.MouseID,palatable_session=p.MouseID,water_age=r.Age,palatable_age=p.Age,
                              age_gap_days=abs(r.Age-p.Age),water_total_g=r.total_g,palatable_total_g=p.total_g))
    pairs=pd.DataFrame(pairs)
    identities=bins[['MouseID','inferred_animal_id']].drop_duplicates().set_index('MouseID').inferred_animal_id
    pairs['water_animal_id']=pairs.water_session.map(identities);pairs['palatable_animal_id']=pairs.palatable_session.map(identities)
    # Age-matching is not biological pairing: some animals reappear in another
    # matched pair. Cluster connected pairs so their contrasts are not independent.
    parents={x:x for x in set(pairs.water_animal_id)|set(pairs.palatable_animal_id)}
    def root(x):
        while parents[x]!=x:x=parents[x]
        return x
    for _,r in pairs.iterrows():parents[root(r.palatable_animal_id)]=root(r.water_animal_id)
    pairs['pair_component']=pairs.water_animal_id.map(root);save_table(pairs,'age_matched_intake_pairs')
    tests=[]
    for sex,g in pairs.groupby('Sex'):
        g=g.copy();g['difference']=g.water_total_g-g.palatable_total_g
        fit=smf.ols('difference ~ 1',g).fit(cov_type='cluster',cov_kwds={'groups':g.pair_component},use_t=True)
        tests.append(dict(Sex=sex,n_pairs=len(g),n_independent_components=g.pair_component.nunique(),t=fit.tvalues.Intercept,p=fit.pvalues.Intercept,mean_age_gap_days=g.age_gap_days.mean(),max_age_gap_days=g.age_gap_days.max(),test='exploratory mean matched difference; connected-animal-component robust t'))
    tests=pd.DataFrame(tests);tests['q']=bh(tests.p);save_table(tests,'age_matched_intake_tests')

def sensitivity_analysis(bins):
    rows=[]
    pal=bins[bins.modeling_cohort]
    for threshold in [0,.05,.1,.25,.5,1.]:
        rs=[];ps=[]
        for mid,g in pal[pal.total_g>threshold].groupby('MouseID'):
            if len(g)>=6 and g.preference.std()>1e-12 and g.total_g.std()>1e-12:
                r,p=stats.pearsonr(g.total_g,g.preference);rs.append(r);ps.append(p)
        rows.append(dict(min_total_g=threshold,n_eligible=len(ps),median_r=np.median(rs),n_nominal_sig=sum(np.array(ps)<.05),n_FDR_sig=sum(bh(ps)<.05)))
    save_table(pd.DataFrame(rows),'low_intake_sensitivity')
    # Four models fit the observed proportions directly. No latent trial counts.
    rows=[];pred=[]
    for mid,g in pal.groupby('MouseID'):
        g=g.dropna(subset=['preference']).sort_values('Time_start');y=g.preference.to_numpy();t=g.ZT.to_numpy();elapsed=g.elapsed_h.to_numpy();w=g.total_g.to_numpy();w=w/w.mean()
        design={'constant':np.ones((len(g),1)),'linear_drift':np.column_stack([np.ones(len(g)),(elapsed-12)/12]),
                'day_night_step':np.column_stack([np.ones(len(g)),(t>=12).astype(float)]),
                '24h_harmonic':np.column_stack([np.ones(len(g)),np.cos(t*np.pi/12),np.sin(t*np.pi/12)])}
        for name,X in design.items():
            squared=[];weighted=[]
            for j in range(len(y)):
                train=np.arange(len(y))!=j;b=np.linalg.lstsq(X[train]*np.sqrt(w[train])[:,None],y[train]*np.sqrt(w[train]),rcond=None)[0]
                phat=float(np.clip(X[j]@b,0,1));squared.append((y[j]-phat)**2);weighted.append(w[j]*(y[j]-phat)**2)
                pred.append(dict(MouseID=mid,model=name,bin_index=int(g.iloc[j].bin_index),observed=y[j],predicted=phat,heldout_weight=w[j]))
            rows.append(dict(MouseID=mid,model=name,n_bins=len(y),loo_mse=np.mean(squared),loo_weighted_mse=np.sum(weighted)/np.sum(w)))
    save_table(pd.DataFrame(rows),'aggregate_model_cross_validation');save_table(pd.DataFrame(pred),'aggregate_heldout_predictions')

