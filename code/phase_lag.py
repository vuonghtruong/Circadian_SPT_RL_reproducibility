"""Paired acrophase lags and circular profile cross-correlation.

Positive lag means preference follows total intake. Sessions are averaged within
likely animal before population summaries. No animals are selected by lag sign.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

PERIOD=24.0
SEED=20261027
REPEATS=10000

def wrap(x):return (np.asarray(x)+12)%24-12

def read_csv(base,name):return pd.read_csv(base/'results'/(name+'.csv'))

def animal_vectors(frame):
    q=frame.copy()
    q['lag_real']=np.cos(q.lag_h*np.pi/12)
    q['lag_imag']=np.sin(q.lag_h*np.pi/12)
    units=q.groupby('inferred_animal_id',sort=True).agg(lag_real=('lag_real','mean'),lag_imag=('lag_imag','mean'),n_sessions=('MouseID','size'),Sex=('Sex','first'))
    units['lag_h']=np.angle(units.lag_real+1j*units.lag_imag)*12/np.pi
    units['within_animal_R']=np.hypot(units.lag_real,units.lag_imag)
    assert q.groupby('inferred_animal_id').Sex.nunique().max()==1
    return units.reset_index()

def summarize(units,scope,repeats=REPEATS):
    z=units.lag_real.to_numpy()+1j*units.lag_imag.to_numpy()
    if len(z)<3:return dict(scope=scope,n_animals=len(z),n_sessions=int(units.n_sessions.sum()),mean_lag_h=np.nan,ci_low_h=np.nan,ci_high_h=np.nan,p_exchange=np.nan,locking_R=np.nan),pd.DataFrame()
    rg=np.random.default_rng(SEED+int(hashlib.sha256(scope.encode()).hexdigest()[:8],16))
    mu=np.angle(z.mean())*12/np.pi
    idx=rg.integers(0,len(z),size=(repeats,len(z)))
    bm=np.angle(z[idx].mean(axis=1))*12/np.pi
    low,high=np.quantile(wrap(bm-mu),[.025,.975])+mu
    signs=rg.choice([-1,1],size=(repeats,len(z)))
    observed=abs(z.imag.mean())
    p=(1+np.sum(abs((signs*z.imag).mean(axis=1))>=observed-1e-14))/(repeats+1)
    return dict(scope=scope,n_animals=len(z),n_sessions=int(units.n_sessions.sum()),mean_lag_h=mu,ci_low_h=low,ci_high_h=high,p_exchange=p,locking_R=abs(z.mean()),bootstrap_repeats=repeats,method='mean session lag phasor per animal; animal bootstrap; centered circular-percentile interval; animal-level intake/preference exchange test'),pd.DataFrame(dict(scope=scope,replicate=np.arange(repeats),mean_lag_h=bm))

def cross_corr(x,y):
    """r(total[t], preference[t+lag]), periodic at24h; no zero padding."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
    if den<=1e-14:return np.full(8,np.nan)
    return np.array([np.dot(x,np.roll(y,-k))/den for k in range(-4,4)])

def main(base,dest):
    dest.mkdir(parents=True,exist_ok=True)
    phase=read_csv(base,'paired_phases')
    phase['lag_h']=wrap(phase.pref_phase-phase.total_phase)
    primary=phase[phase.legacy_age_cohort].dropna(subset=['total_phase','pref_phase']).copy()
    units=animal_vectors(primary)
    primary.to_csv(dest/'figure2F_paired_lag_sessions.csv',index=False)
    units.to_csv(dest/'figure2F_paired_lag_animals.csv',index=False)
    bins=pd.read_csv(base/'data/analysis_bins.csv')
    dates=bins.groupby('MouseID').Time_start.min()
    earliest=primary.assign(session_start=primary.MouseID.map(dates)).sort_values(['session_start','MouseID']).drop_duplicates('inferred_animal_id')
    scopes=[('Control_age_all',units),('Control_age_F',units[units.Sex.eq('F')]),('Control_age_M',units[units.Sex.eq('M')]),
        ('Control_age_earliest',animal_vectors(earliest)),
        ('Control_age_both_nominal',animal_vectors(primary[(primary.total_p<.05)&(primary.pref_p<.05)])),
        ('Modeling_all',animal_vectors(phase[phase.modeling_cohort].dropna(subset=['total_phase','pref_phase'])))]
    summary=[];boots=[]
    for name,g in scopes:
        s,b=summarize(g,name);summary.append(s)
        if len(b):boots.append(b)
    summary=pd.DataFrame(summary)
    # Preserve the three display-group family (pooled, female, male).
    # BH FDR adjusts these exploratory exchange tests; other scopes stay raw.
    from statsmodels.stats.multitest import multipletests
    p=summary.loc[:2,'p_exchange'].to_numpy()
    summary['q_bh_three_groups']=np.nan
    summary.loc[:2,'q_bh_three_groups']=multipletests(p,method='fdr_bh')[1]
    summary.to_csv(dest/'figure2F_paired_lag_summary.csv',index=False)
    pd.concat(boots,ignore_index=True).to_csv(dest/'figure2F_paired_lag_bootstrap.csv',index=False)
    # Independent cosinor reconstruction validates phase convention and lag sign.
    recovered=[];xc=[];excluded=[]
    for mid,g in bins[bins.legacy_age_cohort].groupby('MouseID'):
        g=g.sort_values('bin_index');t=g.ZT.to_numpy(float)
        params=[]
        for metric in ['total_g','preference']:
            yy=g[metric].to_numpy(float);ok=np.isfinite(yy)&np.isfinite(t)
            design=np.column_stack([np.ones(ok.sum()),np.cos(t[ok]*np.pi/12),np.sin(t[ok]*np.pi/12)])
            coef=np.linalg.lstsq(design,yy[ok],rcond=None)[0]
            params.append(np.arctan2(coef[2],coef[1])*12/np.pi%24)
        recovered.append(dict(MouseID=mid,total_phase_check=params[0],pref_phase_check=params[1]))
        x=g.total_g.to_numpy(float);y=g.preference.to_numpy(float)
        if len(g)!=8 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.std(x)<1e-12 or np.std(y)<1e-12:
            excluded.append(dict(MouseID=mid,reason='incomplete or constant 8-bin profile'));continue
        correlations=cross_corr(x,y)
        assert np.allclose(correlations,[np.corrcoef(x,np.roll(y,-k))[0,1] for k in range(-4,4)],atol=1e-12)
        assert abs(correlations.mean())<1e-12 and np.max(abs(correlations))<=1+1e-12
        for lag,r in zip(np.arange(-12,12,3),correlations):
            xc.append(dict(MouseID=mid,inferred_animal_id=g.inferred_animal_id.iloc[0],Sex=g.Sex.iloc[0],lag_h=int(lag),r=float(r)))
    independent=primary.merge(pd.DataFrame(recovered),on='MouseID',validate='one_to_one')
    max_phase_error=max(abs(wrap(independent.total_phase-independent.total_phase_check)).max(),abs(wrap(independent.pref_phase-independent.pref_phase_check)).max())
    assert max_phase_error<1e-8,max_phase_error
    xcs=pd.DataFrame(xc);xca=xcs.groupby(['inferred_animal_id','Sex','lag_h'],as_index=False).r.mean()
    xcs.to_csv(dest/'intake_preference_xcorr_sessions.csv',index=False);xca.to_csv(dest/'intake_preference_xcorr_animals.csv',index=False)
    pd.DataFrame(excluded,columns=['MouseID','reason']).to_csv(dest/'intake_preference_xcorr_exclusions.csv',index=False)
    curves=[];peak=[];contrasts=[]
    lags=np.arange(-12,12,3)
    for name,g in [('All',xca),('F',xca[xca.Sex.eq('F')]),('M',xca[xca.Sex.eq('M')])]:
        wide=g.pivot(index='inferred_animal_id',columns='lag_h',values='r').reindex(columns=lags)
        z=wide.to_numpy();rg=np.random.default_rng(SEED+len(z));idx=rg.integers(0,len(z),size=(REPEATS,len(z)))
        m=z.mean(axis=0);b=z[idx].mean(axis=1);lo,hi=np.quantile(b,[.025,.975],axis=0)
        maxdev=np.quantile(np.max(abs(b-m),axis=1),.95)
        difference=m-m[4];db=b-b[:,[4]]
        difference_maxdev=np.quantile(np.max(abs(db-difference),axis=1),.95)
        for j,lag in enumerate(lags):
            dl,dh=np.quantile(db[:,j],[.025,.975])
            contrasts.append(dict(scope=name,lag_h=int(lag),r_difference_vs_zero_lag=difference[j],ci_low=dl,ci_high=dh,simultaneous_low=difference[j]-difference_maxdev,simultaneous_high=difference[j]+difference_maxdev,n_animals=len(z)))
        for j,lag in enumerate(lags):curves.append(dict(scope=name,lag_h=int(lag),mean_r=m[j],ci_low=lo[j],ci_high=hi[j],simultaneous_low=max(-1,m[j]-maxdev),simultaneous_high=min(1,m[j]+maxdev),n_animals=len(z),n_sessions=xcs[xcs.inferred_animal_id.isin(wide.index)].MouseID.nunique()))
        maximum=m.max();best=lags[np.isclose(m,maximum,atol=1e-12)]
        best_b=lags[b.argmax(axis=1)]
        for lag in lags:peak.append(dict(scope=name,lag_h=int(lag),bootstrap_peak_fraction=np.mean(best_b==lag),observed_peak=bool(lag in best)))
    pd.DataFrame(curves).to_csv(dest/'intake_preference_xcorr_summary.csv',index=False)
    pd.DataFrame(peak).to_csv(dest/'intake_preference_xcorr_peak_bootstrap.csv',index=False)
    pd.DataFrame(contrasts).to_csv(dest/'intake_preference_xcorr_vs_zero.csv',index=False)
    # Known 3h shifted waveform establishes positive-lag convention.
    t=np.arange(8)*3;xx=np.cos(2*np.pi*t/24);yy=np.cos(2*np.pi*(t-3)/24)
    assert lags[cross_corr(xx,yy).argmax()]==3
    assert abs(wrap(1-23)-2)<1e-14
    # Equal-animal calculation is invariant when every session is duplicated.
    doubled=animal_vectors(pd.concat([primary,primary],ignore_index=True))
    assert np.allclose(doubled[['lag_real','lag_imag']],units[['lag_real','lag_imag']],atol=1e-14)
    report=dict(seed=SEED,bootstrap_repeats=REPEATS,primary_n_animals=len(units),primary_n_sessions=len(primary),positive_lag='preference later than total intake',phase_definition='wrap(preference acrophase - intake acrophase) in [-12,12) hours',cross_correlation_definition='Pearson r(total[t],preference[t+lag]); periodic8bin profiles; positive lag preference follows intake',cross_correlation_resolution_h=3,xcorr_excluded_sessions=excluded,independent_phase_reconstruction_max_error_h=float(max_phase_error),independent_pearson_calculation_check=True,synthetic_positive_lag_check=True,repeated_session_weight_check=True,bootstrap_scope='animal sampling conditional on estimated session phases; does not propagate full cosinor parameter uncertainty',limitations=['one24h cycle does not establish endogenous period','weak harmonics yield uncertain acrophases','circular profile correlation assumes periodic continuation','preference contains total intake in its denominator','exploratory replacement of previous panel F; original phase-concentration results retained'])
    (dest/'phase_lag_manifest.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string(index=False));print(pd.DataFrame(curves).query("scope=='All'").to_string(index=False));print(pd.DataFrame(contrasts).query("scope=='All' and lag_h==3").to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();main(a.base.resolve(),a.dest.resolve())
