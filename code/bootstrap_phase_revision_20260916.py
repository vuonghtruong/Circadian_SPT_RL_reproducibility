"""Paired animal bootstrap of phase concentration, not a test across resamples."""
from common import *
import json

def build(repeats=5000):
    frame=pd.read_csv(OUT/'results/paired_phases.csv')
    frame=frame[frame.legacy_age_cohort].dropna(subset=['total_phase','pref_phase'])
    # Each animal contributes the mean of its session phasors. Cluster sizes
    # therefore do not give repeatedly tested animals greater population weight.
    frame['intake_cos']=np.cos(frame.total_phase*np.pi/12);frame['intake_sin']=np.sin(frame.total_phase*np.pi/12)
    frame['pref_cos']=np.cos(frame.pref_phase*np.pi/12);frame['pref_sin']=np.sin(frame.pref_phase*np.pi/12)
    units=frame.groupby('inferred_animal_id')[['intake_cos','intake_sin','pref_cos','pref_sin']].mean().sort_index()
    units['n_sessions']=frame.groupby('inferred_animal_id').size()
    save_table(units.reset_index(),'figure2F_bootstrap_animal_units')
    z=units[['intake_cos','intake_sin','pref_cos','pref_sin']].to_numpy()
    rg=rng_for('paired_animal_phase_concentration');idx=rg.integers(0,len(z),size=(repeats,len(z)))
    means=z[idx].mean(axis=1);a=np.hypot(means[:,0],means[:,1]);b=np.hypot(means[:,2],means[:,3]);difference=a-b
    point=z.mean(axis=0);ra=np.hypot(point[0],point[1]);rb=np.hypot(point[2],point[3])
    rows=pd.DataFrame(dict(replicate=np.arange(repeats),intake_R=a,preference_R=b,difference_R_intake_minus_preference=difference))
    save_table(rows,'figure2F_phase_bootstrap')
    summary=[]
    for metric,estimate,v in [('intake_R',ra,a),('preference_R',rb,b),('difference_R_intake_minus_preference',ra-rb,difference)]:
        lo,hi=np.quantile(v,[.025,.975]);summary.append(dict(metric=metric,estimate=estimate,ci_low=lo,ci_high=hi,n_animals=len(z),n_sessions=len(frame),bootstrap_repeats=repeats,method='paired animal-cluster bootstrap; one mean session phasor per animal; percentile 95% interval'))
    save_table(pd.DataFrame(summary),'figure2F_phase_bootstrap_summary')
    # Keep the originally requested individual-R2 analysis, with its bootstrap
    # uncertainty, even though the new main panel asks the phase-coherence question.
    r2=pd.read_csv(OUT/'results/figure2F_individual_R2.csv');animal_ids=sorted(r2.inferred_animal_id.unique())
    sufficient=[]
    for mid in animal_ids:
        g=r2[r2.inferred_animal_id.eq(mid)];x=g.total_g.to_numpy();y=g.preference.to_numpy()
        sufficient.append([len(g),x.sum(),y.sum(),(x*x).sum(),(y*y).sum(),(x*y).sum()])
    u=np.asarray(sufficient);ri=rng_for('individual_R2_cluster_bootstrap').integers(0,len(u),size=(repeats,len(u)));s=u[ri].sum(axis=1)
    denominator=np.sqrt((s[:,3]-s[:,1]**2/s[:,0])*(s[:,4]-s[:,2]**2/s[:,0]));corr=(s[:,5]-s[:,1]*s[:,2]/s[:,0])/denominator
    save_table(pd.DataFrame(dict(replicate=np.arange(repeats),pearson_r=corr)),'individual_R2_cluster_bootstrap')
    lo,hi=np.quantile(corr,[.025,.975]);test=pd.read_csv(OUT/'results/figure2F_tests.csv').iloc[0]
    save_table(pd.DataFrame([dict(estimate=test.r,ci_low=lo,ci_high=hi,n_animals=len(u),n_sessions=len(r2),cluster_slope_p=test.cluster_slope_p,bootstrap_repeats=repeats)]),'individual_R2_cluster_bootstrap_summary')
    # Independent explicit multiplicity calculation for a nontrivial sample.
    counts=np.bincount(idx[0],minlength=len(z));manual=(z*counts[:,None]).sum(axis=0)/len(z)
    assert np.allclose(manual,means[0],atol=1e-14)
    assert (counts>1).any() and (counts==0).any()
    assert len(units)==frame.inferred_animal_id.nunique() and units.n_sessions.sum()==len(frame)
    assert np.all((rows[['intake_R','preference_R']]>=0)&(rows[['intake_R','preference_R']]<=1))
    (OUT/'results/figure2F_bootstrap_manifest.json').write_text(json.dumps(dict(n_animals=len(z),n_sessions=len(frame),repeats=repeats,multiplicity_check_passed=True,resampled_cluster_rows=len(idx[0]),resamples_are_not_independent_animals=True,no_resample_correlation_p_value=True),indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

