"""Explicit statistical units and comparison families for figure annotations."""
from common import *
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

def fdr_bh(p):return multipletests(np.asarray(p,float),method='fdr_bh')[1]

def partially_paired_phase_test(frame,metric,condition,repeats=9999):
    """Permute within shared animals and among unpaired animals separately."""
    a=frame[frame.Condition_real.eq('Control')].set_index('inferred_animal_id')[metric].dropna()
    b=frame[frame.Condition_real.eq(condition)].set_index('inferred_animal_id')[metric].dropna()
    assert a.index.is_unique and b.index.is_unique
    if len(a)==0 or len(b)==0:raise ValueError('Both condition groups require estimable phases')
    shared=a.index.intersection(b.index);ao=a.index.difference(shared);bo=b.index.difference(shared)
    za=np.exp(1j*a*np.pi/12);zb=np.exp(1j*b*np.pi/12)
    observed=abs(za.mean()-zb.mean());rg=rng_for(metric+condition+'condition_partially_paired')
    pa=za.loc[shared].to_numpy();pb=zb.loc[shared].to_numpy();pool=np.r_[za.loc[ao],zb.loc[bo]]
    null=np.empty(repeats)
    for j in range(repeats):
        swap=rg.random(len(shared))<.5;pair_a=np.where(swap,pb,pa);pair_b=np.where(swap,pa,pb)
        shuffled=rg.permutation(pool)
        ma=(pair_a.sum()+shuffled[:len(ao)].sum())/len(a);mb=(pair_b.sum()+shuffled[len(ao):].sum())/len(b)
        null[j]=abs(ma-mb)
    return dict(metric=metric,reference='Control',condition=condition,n_reference=len(a),n_condition=len(b),n_shared_animals=len(shared),
        n_reference_only=len(ao),n_condition_only=len(bo),vector_difference=observed,p=(1+sum(null>=observed-1e-14))/(repeats+1),
        test='partially paired label permutation: within-animal swaps plus unpaired label shuffles; phase/concentration jointly')

def compute_panel_statistics(bins):
    # One mean per inferred animal and fluid, so repeat sessions cannot inflate n.
    fluid=bins[bins.Condition_real.eq('Control')&~bins.excluded_AD].groupby(['MouseID','inferred_animal_id','fluid']).preference.mean().reset_index()
    fluid=fluid.groupby(['inferred_animal_id','fluid']).preference.mean().reset_index();save_table(fluid,'figure1B_animal_means')
    rows=[]
    for name,g in fluid.groupby('fluid'):
        v=g.preference.dropna();t,p=stats.ttest_1samp(v,.5);rows.append(dict(fluid=name,n=len(v),mean=v.mean(),sem=stats.sem(v),t=t,p=p,test='two-sided one-sample t versus 0.5; animal means'))
    r=pd.DataFrame(rows);r['q_bh']=fdr_bh(r.p);save_table(r,'figure1B_tests')
    rows=[];water=fluid[fluid.fluid.eq('water')].preference.dropna()
    for name in ['glucose','sucrose','saccharin']:
        selected=fluid[fluid.fluid.isin(['water',name])].copy();selected['contrast']=selected.fluid.eq(name).astype(float)
        fit=sm.OLS(selected.preference,sm.add_constant(selected.contrast)).fit(cov_type='cluster',cov_kwds={'groups':selected.inferred_animal_id},use_t=True)
        v=selected[selected.fluid.eq(name)].preference;overlap=set(selected.loc[selected.fluid.eq(name),'inferred_animal_id'])&set(selected.loc[selected.fluid.eq('water'),'inferred_animal_id'])
        t,p=(fit.tvalues.contrast,fit.pvalues.contrast) if overlap else stats.ttest_ind(v,water,equal_var=False)
        method='animal-cluster-robust OLS t' if overlap else 'Welch t (independent animals)'
        rows.append(dict(fluid=name,reference='water',n=len(v),n_reference=len(water),n_animals=selected.inferred_animal_id.nunique(),n_shared_animals=len(overlap),difference=fit.params.contrast,t=t,p=p,test='two-sided difference in means; '+method+'; one mean per animal and fluid'))
    r=pd.DataFrame(rows);r['q_bh']=fdr_bh(r.p);save_table(r,'figure1B_vs_water_tests')
    age=bins[bins.legacy_age_cohort]
    pairs=[];rows=[]
    for metric in METRICS:
        a=age.groupby(['MouseID','inferred_animal_id','phase'])[metric].mean().reset_index()
        a=a.groupby(['inferred_animal_id','phase'])[metric].mean().unstack().dropna();d=a.Night-a.Day;t,p=stats.ttest_rel(a.Day,a.Night)
        pairs.append(a.reset_index().assign(metric=metric));rows.append(dict(metric=metric,n=len(a),day_mean=a.Day.mean(),night_mean=a.Night.mean(),day_sem=a.Day.sem(),night_sem=a.Night.sem(),difference=d.mean(),t=t,p=p))
    save_table(pd.concat(pairs),'figure2C_animal_pairs');r=pd.DataFrame(rows);r['q_bh']=fdr_bh(r.p);save_table(r,'figure2C_tests')
    phases=pd.read_csv(OUT/'results/paired_phases.csv');rows=[];curves=[]
    for metric in ['total_phase','pref_phase']:
        for sex,g in phases[phases.legacy_age_cohort].groupby('Sex'):
            g=g.dropna(subset=[metric,'Age']);fit=sm.OLS(g[metric],sm.add_constant(g.Age)).fit(cov_type='cluster',cov_kwds={'groups':g.inferred_animal_id})
            x=np.linspace(g.Age.min(),g.Age.max(),100);ci=fit.get_prediction(sm.add_constant(x)).summary_frame()
            curves.append(pd.DataFrame(dict(metric=metric,Sex=sex,Age=x,mean=ci['mean'],low=ci.mean_ci_lower,high=ci.mean_ci_upper)))
            # Circular-linear association through multiple correlation with sin/cos,
            # assessed by permuting animal-level ages (one earliest age session).
            h=g.sort_values(['Age','MouseID']).drop_duplicates('inferred_animal_id');xx=h.Age.to_numpy();theta=h[metric].to_numpy()*np.pi/12
            design=np.column_stack([np.ones(len(h)),np.sin(theta),np.cos(theta)]);hat=design@np.linalg.pinv(design)
            def score(a):return 1-np.sum((a-hat@a)**2)/np.sum((a-a.mean())**2)
            observed=score(xx);rg=rng_for(metric+sex+'circular_age');null=np.array([score(rg.permutation(xx)) for _ in range(1999)])
            rows.append(dict(metric=metric,Sex=sex,n_sessions=len(g),n_animals=g.inferred_animal_id.nunique(),linear_slope_h_per_day=fit.params.Age,linear_cluster_p=fit.pvalues.Age,circular_linear_R2=observed,circular_permutation_p=(1+sum(null>=observed))/2000,interpretation='straight line is descriptive on 0–24 scale; circular sensitivity uses earliest age session'))
    save_table(pd.concat(curves),'figure2DE_trend_lines');r=pd.DataFrame(rows);r['circular_q_bh']=fdr_bh(r.circular_permutation_p);save_table(r,'figure2DE_tests')
    ind=pd.read_csv(OUT/'results/individual_cosinor.csv');r2=ind[ind.legacy_age_cohort].pivot(index='MouseID',columns='metric',values='r2').dropna(subset=['total_g','preference']).reset_index()
    r2=r2.merge(age[['MouseID','inferred_animal_id']].drop_duplicates(),on='MouseID',validate='one_to_one')
    fit=sm.OLS(r2.preference,sm.add_constant(r2.total_g)).fit(cov_type='cluster',cov_kwds={'groups':r2.inferred_animal_id});rr,pp=stats.pearsonr(r2.total_g,r2.preference)
    x=np.linspace(0,1,100);ci=fit.get_prediction(sm.add_constant(x)).summary_frame();save_table(pd.DataFrame(dict(x=x,mean=ci['mean'],low=ci.mean_ci_lower,high=ci.mean_ci_upper)),'figure2F_trend_line')
    save_table(r2,'figure2F_individual_R2');save_table(pd.DataFrame([dict(n_sessions=len(r2),n_animals=r2.inferred_animal_id.nunique(),r=rr,pearson_p=pp,cluster_slope_p=fit.pvalues.total_g,slope=fit.params.total_g)]),'figure2F_tests')
    # Retain the earliest observation within each condition. A later PA session
    # remains informative even when that animal had an earlier Control session.
    meta=bins.drop_duplicates('MouseID');eligible=meta[meta.palatable_cohort & meta.Condition_real.isin(['Control','PA','RF-morning','RF-afternoon'])].sort_values(['experiment_date','MouseID']).drop_duplicates(['inferred_animal_id','Condition_real'])
    ph=phases[phases.MouseID.isin(eligible.MouseID)];save_table(ph,'figure2IJ_phase_points');rows=[]
    for metric in ['total_phase','pref_phase']:
        for condition in ['PA','RF-morning','RF-afternoon']:
            rows.append(partially_paired_phase_test(ph,metric,condition))
    r=pd.DataFrame(rows);r['q_bh']=fdr_bh(r.p);save_table(r,'figure2IJ_tests')
    dn=pd.read_csv(OUT/'results/illustrative_day_night.csv');rows=[]
    for model,g in dn.groupby('model'):
        a=g.pivot(index='run',columns='phase',values='preference');t,p=stats.ttest_rel(a.Day,a.Night);rows.append(dict(model=model,n_runs=len(a),t=t,p=p,night_minus_day=(a.Night-a.Day).mean(),test='paired simulated runs; no empirical animal inference'))
    r=pd.DataFrame(rows);r['q_bh']=fdr_bh(r.p);save_table(r,'figure3A_tests')
    print('Panel annotation statistics complete',flush=True)

