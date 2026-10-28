"""Numerically replay every saved primary fit; verify cohorts and figure statistics."""
from common import *
from models import MODEL_NAMES, PARAM_NAMES, allocate, predict
from statistics import cosinor
import json, importlib.metadata

def main():
    records=[]
    def check(name,condition,detail=None):
        if not bool(condition):raise AssertionError((name,detail))
        records.append(dict(check=name,passed=True,detail=detail))
    b=pd.read_csv(ROOT/'data/analysis_bins.csv');primary=b[b.modeling_cohort]
    fits=pd.read_csv(ROOT/'results/corrected_expected_model_fits.csv')
    saved=pd.read_csv(ROOT/'results/corrected_expected_model_predictions.csv').set_index(['MouseID','model','bin_index'])
    check('245 sessions and 1960 nominal intervals',b.MouseID.nunique()==245 and len(b)==1960)
    check('No duplicated interval keys',not b.duplicated(['MouseID','bin_index']).any())
    check('Eight chronological intervals per session',b.groupby('MouseID').size().eq(8).all())
    check('No AD animals in shared dataframe',not b.excluded_AD.any())
    check('140 distinct likely animals in primary cohort',primary.MouseID.nunique()==140 and primary.inferred_animal_id.nunique()==140)
    check('No restricted-feeding animals in model cohort',not primary.excluded_RF.any())
    check('Model cohort is palatable only',primary.fluid.isin(['sucrose','saccharin']).all())
    check('Total intake reconstructs from two bottle columns',np.allclose(b.total_g,b.bottle_A_Consumption+b.bottle_B_Consumption,atol=1e-12,rtol=0))
    positive=b.total_g>0
    check('Preference reconstructs from reward/total loss',np.allclose(b.loc[positive,'preference'],b.loc[positive,'reward_g']/b.loc[positive,'total_g'],atol=1e-12,rtol=0))
    check('Missing ratios retained where denominator is zero',b.loc[~positive,'preference'].isna().all())
    check('Every animal has all ten fitted candidates',len(fits)==1400 and fits.groupby('MouseID').model.nunique().eq(10).all())
    worst_prediction=0.;worst_loss=0.;replayed=[]
    for mid,g in primary.groupby('MouseID'):
        g=g.sort_values('Time_start');n=allocate(g.total_g.to_numpy(float),1000)
        zt=g.ZT.to_numpy(float);elapsed=g.elapsed_h.to_numpy(float);idx=g.bin_index.to_numpy(np.int64)
        y=g.preference.to_numpy(float);w=g.total_g.to_numpy(float);ok=np.isfinite(y)&(w>0);ww=w[ok]/w[ok].sum()
        for row in fits[fits.MouseID.eq(mid)].itertuples():
            par=np.asarray(json.loads(row.parameters));p=predict(row.model,par,zt,elapsed,idx,n)
            old=saved.loc[(mid,row.model)].reindex(idx)
            worst_prediction=max(worst_prediction,float(np.max(abs(p-old.predicted.to_numpy()))))
            pp=np.clip(p[ok],1e-10,1-1e-10);ce=-np.sum(ww*(y[ok]*np.log(pp)+(1-y[ok])*np.log1p(-pp)))
            worst_loss=max(worst_loss,abs(ce-row.cross_entropy))
            check_n=np.array_equal(old.N_bin.to_numpy(),n)
            if not check_n:raise AssertionError((mid,row.model,'latent allocation changed'))
            if row.k!=len(par):raise AssertionError((mid,row.model,'parameter count'))
            for N in [100,1000,10000]:
                if abs(getattr(row,'pseudo_AIC_'+str(N))-(2*N*ce+2*len(par)))>1e-7:
                    raise AssertionError((mid,row.model,N,'pseudo score'))
            replayed.extend(dict(MouseID=mid,model=row.model,bin_index=int(i),predicted=float(x)) for i,x in zip(idx,p))
    check('Replay all 11200 fitted bin predictions',worst_prediction<1e-10,dict(max_absolute_error=worst_prediction,n_predictions=len(replayed)))
    check('Replay all 1400 weighted cross-entropy objectives',worst_loss<1e-10,dict(max_absolute_error=worst_loss))
    eight=fits[~fits.model.isin(['linear_bias','step_bias'])]
    counts=eight.loc[eight.groupby('MouseID').pseudo_AIC_1000.idxmin()].model.value_counts().to_dict()
    check('Eight-model winners reproduce exactly',counts=={'full':73,'full_no_swap':37,'reward':16,'reward_no_swap':10,'bias':3,'simple_no_swap':1},counts)
    cv=pd.read_csv(ROOT/'results/variance_extended_blocked_cv.csv')
    check('Matched 140 animals x 10 models x 4 folds',len(cv)==5600 and not cv.duplicated(['MouseID','model','fold']).any() and cv.groupby(['MouseID','model']).size().eq(4).all())
    cvmean=cv.groupby(['MouseID','model']).heldout_weighted_mse.mean().groupby('model').mean()
    check('Bias has lowest matched held-out mean error',cvmean.idxmin()=='bias',cvmean.sort_values().to_dict())
    # Independently reproduce temporal-variance capture without using its summary.
    prediction=pd.DataFrame(replayed).merge(primary[['MouseID','bin_index','total_g','preference']],on=['MouseID','bin_index'],validate='many_to_one')
    metrics=[]
    for (mid,model),g in prediction.groupby(['MouseID','model']):
        q=g.preference.notna()&g.total_g.gt(0);y=g.loc[q,'preference'].to_numpy();w=g.loc[q,'total_g'].to_numpy();w=w/w.sum()
        metrics.append(dict(MouseID=mid,model=model,mse=np.sum(w*(y-g.loc[q,'predicted'].to_numpy())**2),variance=np.sum(w*(y-np.sum(w*y))**2)))
    m=pd.DataFrame(metrics).groupby('model')[['mse','variance']].mean();r2=1-m.mse/m.variance
    expected=pd.read_csv(ROOT/'results/variance_model_summary.csv').set_index('model').pooled_within_R2
    check('Temporal variance captured recomputes from fitted profiles',np.allclose(r2,expected.reindex(r2.index),atol=1e-10,rtol=0),r2.to_dict())
    # Inferential unit for F is the animal, not each bootstrap replicate.
    phase=pd.read_csv(ROOT/'results/paired_phases.csv');phase=phase[phase.legacy_age_cohort].dropna(subset=['total_phase','pref_phase'])
    z=np.column_stack([np.cos(phase.total_phase*np.pi/12),np.sin(phase.total_phase*np.pi/12),np.cos(phase.pref_phase*np.pi/12),np.sin(phase.pref_phase*np.pi/12)])
    units=pd.DataFrame(z,index=phase.inferred_animal_id).groupby(level=0).mean().sort_index();point=units.mean().to_numpy()
    estimate=np.array([np.hypot(*point[:2]),np.hypot(*point[2:])]);fs=pd.read_csv(ROOT/'results/figure2F_phase_bootstrap_summary.csv').set_index('metric')
    check('Archived phase-concentration analysis uses 113 animal units across 152 sessions',len(units)==113 and len(phase)==152 and np.allclose(estimate,fs.loc[['intake_R','preference_R'],'estimate'],atol=1e-12))
    d=fs.loc['difference_R_intake_minus_preference'];check('Phase-concentration difference interval includes zero',d.ci_low<0<d.ci_high)
    # Current Figure2F: independent reconstruction from the shared raw profiles.
    from phase_lag import animal_vectors,cross_corr
    lag_units=animal_vectors(phase)
    lag_summary=pd.read_csv(ROOT/'results/figure2F_paired_lag_summary.csv').set_index('scope')
    lag_mu=np.angle(np.mean(lag_units.lag_real+1j*lag_units.lag_imag))*12/np.pi
    check('Paired lag matches 113 equal animal units',len(lag_units)==113 and abs(lag_mu-lag_summary.loc['Control_age_all','mean_lag_h'])<1e-12)
    interval=lag_summary.loc['Control_age_all'];check('Paired lag uncertainty includes zero',interval.ci_low_h<0<interval.ci_high_h)
    curves=[]
    for mid,g in b[b.legacy_age_cohort].groupby('MouseID'):
        g=g.sort_values('bin_index');x=g.total_g.to_numpy(float);y=g.preference.to_numpy(float)
        if np.isfinite(x).all() and np.isfinite(y).all() and np.std(x)>1e-12 and np.std(y)>1e-12:
            for lag,r in zip(range(-12,12,3),cross_corr(x,y)):curves.append(dict(inferred_animal_id=g.inferred_animal_id.iloc[0],lag_h=lag,r=r,MouseID=mid))
    cc=pd.DataFrame(curves);mean=cc.groupby(['inferred_animal_id','lag_h']).r.mean().groupby('lag_h').mean()
    saved_curve=pd.read_csv(ROOT/'results/intake_preference_xcorr_summary.csv');saved_curve=saved_curve[saved_curve.scope.eq('All')].set_index('lag_h')
    check('Current Figure2F reconstructs all eight periodic profile correlations',np.allclose(mean,saved_curve.loc[mean.index,'mean_r'],atol=1e-12))
    check('Cross-correlation uses 149 complete sessions from 113 animals',cc.MouseID.nunique()==149 and cc.inferred_animal_id.nunique()==113)
    check('Cross-correlation maximum is at positive3h lag',int(mean.idxmax())==3)
    contrast=pd.read_csv(ROOT/'results/intake_preference_xcorr_vs_zero.csv').query("scope=='All' and lag_h==3").iloc[0]
    check('Selected peak advantage over zero lag remains uncertain',contrast.simultaneous_low<0<contrast.simultaneous_high)
    # Source metadata explains schedule exclusions; no behavioral phase shifts applied.
    light=pd.read_csv(ROOT/'results/light_dark_mouse_audit.csv');light=light[light.modeling_cohort]
    check('Recorded-day-dominant audit count remains 28',int(light.day_dominant.sum())==28)
    # BH FDR correction replay using an independent SciPy implementation.
    family_file=ROOT/'results/correction_families.json'
    for family in json.loads(family_file.read_text(encoding='utf8')):
        frame=pd.read_csv(ROOT/'results'/(family['table']+'.csv'))
        valid=frame[family['adjusted_column']].notna()
        raw=frame.loc[valid,family['raw_column']].to_numpy(float)
        observed=frame.loc[valid,family['adjusted_column']].to_numpy(float)
        check('BH FDR '+family['table'],np.allclose(observed,stats.false_discovery_control(raw,method='bh'),atol=1e-12,rtol=0))
    entrainment=pd.read_csv(ROOT/'results/figure2IJ_tests.csv')
    afternoon=entrainment[entrainment.metric.eq('pref_phase')&entrainment.condition.eq('RF-afternoon')].iloc[0]
    check('Afternoon RF preference BH q is 0.0258',abs(afternoon.q_bh-.0258)<1e-12)
    # Genotype S1 phase vectors and shortest circular offsets.
    gp=pd.read_csv(ROOT/'results/genotype_phase_inclusions.csv')
    gt=pd.read_csv(ROOT/'results/genotype_vs_control_phase_tests.csv')
    check('Pooled genotype S1 has four comparisons',len(gt)==4)
    check('Genotype S1 Control counts 22 in LD and DD',gp[gp.Genotype.eq('Control')].groupby('lighting').inferred_animal_id.nunique().to_dict()=={'DD':22,'LD':22})
    check('Pooled genotype S1 has 44 Control, 17 AB and 22 VB animals',gp.groupby('Genotype').inferred_animal_id.nunique().to_dict()=={'AB':17,'Control':44,'VB':22})
    check('No repeated animal in pooled genotype comparison',gp.inferred_animal_id.is_unique)
    for row in gt.itertuples():
        a=gp[gp.Genotype.eq(row.genotype)][row.metric].to_numpy()
        b=gp[gp.Genotype.eq('Control')][row.metric].to_numpy()
        za=np.exp(1j*a*np.pi/12).mean();zb=np.exp(1j*b*np.pi/12).mean()
        distance=abs(za-zb)
        delta=(np.angle(za)*12/np.pi-np.angle(zb)*12/np.pi+12)%24-12
        check('Genotype phase vector '+row.lighting+' '+row.metric+' '+row.genotype,abs(distance-row.vector_difference)<1e-12 and abs(delta-row.mean_difference_h)<1e-12)
    manifest=ROOT/'manifest_sha256.json'
    if manifest.exists():
        entries=json.loads(manifest.read_text())['files']
        for name,item in entries.items():
            if name.startswith(('data/','results/','tables/','code/')) or name in ['reproduce.py','requirements.txt']:
                check('SHA256 '+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==item['sha256'])
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    summary={'passed':True,'checks':records,'prediction_replay_sha256':hashlib.sha256(pd.DataFrame(replayed).to_csv(index=False,float_format='%.12g').encode()).hexdigest(),
      'scope':'Saved-parameter numerical replay and cohort/statistical invariants; no optimizer rerun or raw-workbook re-preprocessing.'}
    (out/'numerical_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2));print(f'{len(records)} checks passed')

if __name__=='__main__':main()
