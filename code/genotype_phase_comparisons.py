"""Exploratory genotype–Control phase-vector comparisons pooling LD and DD.

Restrict to sucrose and retain the earliest eligible recording per likely animal
across LD and early DD. BH FDR across 2 genotypes × 2 metrics. A separate
lighting-stratified permutation sensitivity checks the pooled comparisons.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from statsmodels.stats.multitest import multipletests

ROOT=Path(os.environ.get('CIRCADIAN_OUTPUT',Path(__file__).resolve().parents[1])).resolve()
BASE=ROOT
DEST=ROOT/'results'
DEST.mkdir(parents=True,exist_ok=True)
REPEATS=999999
b=pd.read_csv(BASE/'data/analysis_bins.csv')
p=pd.read_csv(BASE/'results/paired_phases.csv')
meta=b.drop_duplicates('MouseID').copy()
mapping={'LD':'LD','AB-LD':'LD','VB-LD':'LD','DD':'DD','AB-DD':'DD','VB-DD':'DD'}
meta['lighting']=meta.Condition.map(mapping)
selected=meta[meta.lighting.notna()&meta.fluid.eq('sucrose')&meta.Condition_real.eq('Control')&~meta.excluded_AD&~meta.excluded_RF].copy()
eligible_count=len(selected)
selected=selected.sort_values(['experiment_date','MouseID']).drop_duplicates(['inferred_animal_id','Genotype'])
columns=['MouseID','inferred_animal_id','Genotype','Sex','Age','lighting','experiment_date']
phase=selected[columns].merge(p[['MouseID','total_phase','pref_phase','total_p','pref_p']],on='MouseID',validate='one_to_one')
assert phase.groupby('inferred_animal_id').Genotype.nunique().max()==1
assert phase.inferred_animal_id.is_unique
phase.to_csv(DEST/'genotype_phase_inclusions.csv',index=False)

def mean_hour(z):return np.angle(z.mean())*12/np.pi%24
rows=[]
for lighting in ['LD+DD']:
    for metric in ['total_phase','pref_phase']:
        for genotype in ['AB','VB']:
            c=phase[phase.Genotype.eq('Control')].dropna(subset=[metric])
            g=phase[phase.Genotype.eq(genotype)].dropna(subset=[metric])
            assert c.inferred_animal_id.is_unique and g.inferred_animal_id.is_unique
            assert not set(c.inferred_animal_id)&set(g.inferred_animal_id)
            zc=np.exp(1j*c[metric].to_numpy()*np.pi/12)
            zg=np.exp(1j*g[metric].to_numpy()*np.pi/12)
            observed=abs(zg.mean()-zc.mean());pool=np.r_[zc,zg]
            rg=np.random.default_rng(20261028+int(hashlib.sha256((lighting+metric+genotype).encode()).hexdigest()[:8],16))
            null=np.empty(REPEATS)
            for start in range(0,REPEATS,5000):
                count=min(5000,REPEATS-start)
                shuffled=rg.permuted(np.broadcast_to(pool,(count,len(pool))),axis=1)
                null[start:start+count]=abs(shuffled[:,:len(zc)].mean(1)-shuffled[:,len(zc):].mean(1))
            rows.append(dict(lighting=lighting,metric=metric,genotype=genotype,reference='Control',n_genotype=len(g),n_control=len(c),
                genotype_mean_h=mean_hour(zg),control_mean_h=mean_hour(zc),
                genotype_R=abs(zg.mean()),control_R=abs(zc.mean()),vector_difference=observed,
                p=(1+np.sum(null>=observed-1e-14))/(REPEATS+1),permutations=REPEATS,
                genotype_age_min=g.Age.min(),genotype_age_max=g.Age.max(),control_age_min=c.Age.min(),control_age_max=c.Age.max()))
result=pd.DataFrame(rows)
result['q_BH']=multipletests(result.p,method='fdr_bh')[1]
result['permutation_mc_se']=np.sqrt(result.p*(1-result.p)/(REPEATS+1))
result['mean_difference_h']=(result.genotype_mean_h-result.control_mean_h+12)%24-12
result['mean_direction']=np.where(result.mean_difference_h<0,'earlier','later')
assert np.allclose(result.q_BH,false_discovery_control(result.p,method='bh'),atol=1e-14)
result.to_csv(DEST/'genotype_vs_control_phase_tests.csv',index=False)
summary=phase.groupby('Genotype').agg(n_animals=('inferred_animal_id','nunique'),n_LD=('lighting',lambda x:int(x.eq('LD').sum())),n_DD=('lighting',lambda x:int(x.eq('DD').sum())),n_female=('Sex',lambda x:int(x.eq('F').sum())),n_male=('Sex',lambda x:int(x.eq('M').sum())),age_min=('Age','min'),age_max=('Age','max')).reset_index()
summary.to_csv(DEST/'genotype_phase_cohort_summary.csv',index=False)
report=dict(exploratory=True,statistic='absolute difference between genotype and Control mean complex unit phase vectors; timing and concentration jointly',correction='BH FDR across four pooled tests',sampling_unit='one earliest eligible sucrose recording per likely animal across pooled LD/early DD',permutations=REPEATS,all_estimable_phases_included=True,rhythmicity_p_filter=False,independent_BH_checks_agree=True,eligible_count=eligible_count,selected_count=len(phase),repeated_sessions_removed=eligible_count-len(phase),
    limitations=['This tests phase-vector organization, not equality of the whole intake or preference profile.','Age, sex, batch and lighting proportions are not matched.','Acrophases from weak cosinor fits can be uncertain.','Pooling estimates combined LD/early-DD patterns, not identical effects in each condition.','Nonsignificance does not establish equivalence or absence of genotype effects.'],
    source_sha256={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ['data/analysis_bins.csv','results/paired_phases.csv']})
report['primary_permutation']='Unrestricted animal-label permutation across the pooled LD/early-DD sample'
report['sensitivity']='Lighting-stratified label permutation with a separate BH FDR family of four comparisons'
(DEST/'genotype_phase_analysis_details.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(result[['lighting','metric','genotype','n_genotype','n_control','p','q_BH','genotype_mean_h','control_mean_h']].to_string(index=False))
print(summary.to_string(index=False))

# Sensitivity: preserve the LD/DD allocation of each genotype while permuting
# labels within lighting condition. A separate BH family covers these four tests.
sensitivity=[]
for row in result.itertuples():
    c=phase[phase.Genotype.eq('Control')].dropna(subset=[row.metric])
    g=phase[phase.Genotype.eq(row.genotype)].dropna(subset=[row.metric])
    strata=[]
    for light in ['LD','DD']:
        a=c[c.lighting.eq(light)][row.metric].to_numpy()
        d=g[g.lighting.eq(light)][row.metric].to_numpy()
        strata.append((np.exp(1j*np.r_[a,d]*np.pi/12),len(a)))
    seed=20261029+int(hashlib.sha256((row.metric+row.genotype+'stratified').encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed);null=np.empty(REPEATS)
    for start in range(0,REPEATS,5000):
        count=min(5000,REPEATS-start)
        ca=np.zeros(count,complex);ga=np.zeros(count,complex)
        for values,nc in strata:
            z=rng.permuted(np.broadcast_to(values,(count,len(values))),axis=1)
            ca+=z[:,:nc].sum(1);ga+=z[:,nc:].sum(1)
        null[start:start+count]=abs(ca/len(c)-ga/len(g))
    record=row._asdict();record.pop('Index',None)
    record['p']=(1+np.sum(null>=row.vector_difference-1e-14))/(REPEATS+1)
    sensitivity.append(record)
sensitivity=pd.DataFrame(sensitivity)
sensitivity['q_BH']=multipletests(sensitivity.p,method='fdr_bh')[1]
sensitivity['permutation_mc_se']=np.sqrt(sensitivity.p*(1-sensitivity.p)/(REPEATS+1))
assert np.allclose(sensitivity.q_BH,false_discovery_control(sensitivity.p,method='bh'),atol=1e-14)
sensitivity.to_csv(DEST/'genotype_phase_lighting_stratified_sensitivity.csv',index=False)
print('Lighting-stratified permutation sensitivity')
print(sensitivity[['metric','genotype','mean_difference_h','p','q_BH']].to_string(index=False))
