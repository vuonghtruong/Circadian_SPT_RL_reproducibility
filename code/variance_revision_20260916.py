"""Animal-weighted explained-variance comparisons and complete matched CV.

Saved primary model parameters remain unchanged. Missing no-swap validation
uses the same folds, objective, optimizer and deterministic starts as models.py.
"""
from common import *
import argparse,json,time
from models import predict,allocate,fit_one

MANUSCRIPT=['random','bias','simple','simple_no_swap','reward','reward_no_swap','full','full_no_swap']

def signature():
    paths={'data/analysis_bins.csv':OUT/'data/analysis_bins.csv',
           'code/models.py':Path(__file__).parent/'models.py',
           'results/expected_model_blocked_cv.csv':OUT/'results/expected_model_blocked_cv.csv'}
    return {label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in paths.items()}

def extend_cv(force=False):
    cache=OUT/'results/variance_extended_cv_cache.json';target=OUT/'results/variance_extended_blocked_cv.csv'
    if not force and cache.exists() and target.exists():
        saved=json.loads(cache.read_text())
        if saved['inputs']==signature() and saved['output']==hashlib.sha256(target.read_bytes()).hexdigest():
            print('Extended CV cache verified',flush=True);return pd.read_csv(target)
    b=pd.read_csv(OUT/'data/analysis_bins.csv');rows=[];diagnostics=[];start=time.time()
    for j,(mid,g) in enumerate(b[b.modeling_cohort].groupby('MouseID')):
        g=g.sort_values('Time_start');zt=g.ZT.to_numpy(float);elapsed=g.elapsed_h.to_numpy(float);idx=g.bin_index.to_numpy(np.int64)
        y=g.preference.to_numpy(float);total=g.total_g.to_numpy(float);n=allocate(total,1000)
        for model in ['random','simple_no_swap','reward_no_swap','full_no_swap']:
            for fold in range(4):
                held=idx//2==fold;w=total.copy();w[held]=0
                par,loss,diag=fit_one(model,zt,elapsed,idx,n,y,w,mid+str(fold),starts=3)
                p=predict(model,par,zt,elapsed,idx,n);ok=held&np.isfinite(y)&(total>0)
                if not ok.any():continue
                rows.append(dict(MouseID=mid,model=model,fold=fold,heldout_weighted_mse=float(np.average((y[ok]-p[ok])**2,weights=total[ok])),heldout_mse=float(np.mean((y[ok]-p[ok])**2)),converged=diag['converged']))
                diagnostics.append(dict(MouseID=mid,model=model,fold=fold,parameters=json.dumps(par.tolist()),training_cross_entropy=loss,n_heldout=int(ok.sum()),**diag))
        if (j+1)%10==0:print(f'Extended CV {j+1}/{b.loc[b.modeling_cohort,"MouseID"].nunique()} animals, {time.time()-start:.0f}s',flush=True)
    old=pd.read_csv(OUT/'results/expected_model_blocked_cv.csv')
    result=pd.concat([old,pd.DataFrame(rows)],ignore_index=True).sort_values(['MouseID','model','fold'])
    assert not result.duplicated(['MouseID','model','fold']).any()
    assert result.groupby(['MouseID','model']).size().eq(4).all()
    assert result.groupby('MouseID').model.nunique().eq(10).all()
    save_table(result,'variance_extended_blocked_cv');save_table(pd.DataFrame(diagnostics),'variance_extended_cv_fits')
    cache.write_text(json.dumps(dict(inputs=signature(),output=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
    return result

def calculate(repeats=5000):
    b=pd.read_csv(OUT/'data/analysis_bins.csv');p=pd.read_csv(OUT/'results/corrected_expected_model_predictions.csv')
    primary=b[b.modeling_cohort];d=p.merge(primary[['MouseID','bin_index','total_g','preference']],on=['MouseID','bin_index'],validate='many_to_one')
    rows=[]
    for (mid,model),g in d.groupby(['MouseID','model']):
        valid=g.preference.notna()&g.total_g.gt(0);y=g.loc[valid,'preference'].to_numpy();w=g.loc[valid,'total_g'].to_numpy();w=w/w.sum();pred=g.loc[valid,'predicted'].to_numpy()
        mu=np.sum(w*y);mse=np.sum(w*(y-pred)**2);var=np.sum(w*(y-mu)**2)
        rows.append(dict(MouseID=mid,model=model,n_bins=int(valid.sum()),weighted_mean_preference=mu,weighted_mse=mse,within_animal_variance=var,within_animal_R2=1-mse/var if var>1e-12 else np.nan))
    r=pd.DataFrame(rows);save_table(r,'variance_by_animal_model')
    mse=r.pivot(index='MouseID',columns='model',values='weighted_mse').sort_index();variance=r.groupby('MouseID').within_animal_variance.first().reindex(mse.index).to_numpy()
    assert np.allclose(mse.bias,variance,atol=1e-10)
    assert len(mse)==primary.MouseID.nunique()
    rg=rng_for('variance_animal_bootstrap');indices=rg.integers(0,len(mse),size=(repeats,len(mse)))
    vm=variance[indices].mean(axis=1);boot_mse=mse.to_numpy()[indices].mean(axis=1);den=variance.mean()
    boot_r2=1-boot_mse/vm[:,None];boot_random=boot_mse[:,mse.columns.get_loc('random')]
    reduction=1-boot_mse/boot_random[:,None]
    summary=[]
    for k,model in enumerate(mse.columns):
        ci=np.quantile(boot_r2[:,k],[.025,.975]);cr=np.quantile(reduction[:,k],[.025,.975])
        summary.append(dict(model=model,n_animals=len(mse),weighted_mse=mse[model].mean(),within_animal_variance=den,pooled_within_R2=1-mse[model].mean()/den,R2_low=ci[0],R2_high=ci[1],error_reduction_vs_random=1-mse[model].mean()/mse.random.mean(),random_reduction_low=cr[0],random_reduction_high=cr[1],median_animal_R2=r[r.model.eq(model)].within_animal_R2.median(),bootstrap_repeats=repeats))
    save_table(pd.DataFrame(summary),'variance_model_summary')
    increments=[]
    for simple in ['simple','simple_no_swap']:
        for more in ['reward','reward_no_swap','full','full_no_swap']:
            delta=(mse[simple]-mse[more]).mean()/den
            boot=(boot_mse[:,mse.columns.get_loc(simple)]-boot_mse[:,mse.columns.get_loc(more)])/vm
            lo,hi=np.quantile(boot,[.025,.975])
            increments.append(dict(reference_model=simple,comparison_model=more,additional_R2=delta,low=lo,high=hi,n_animals=len(mse),interpretation='descriptive in-sample improvement; models fitted to these observations'))
    save_table(pd.DataFrame(increments),'variance_incremental_comparisons')
    # A post-fit choice within each family is optimistic and is labelled as such.
    fits=pd.read_csv(OUT/'results/corrected_expected_model_fits.csv');families=[]
    for name,models in [('Simple (best switch variant)',['simple','simple_no_swap']),('Reward (best switch variant)',['reward','reward_no_swap']),('Full (best switch variant)',['full','full_no_swap'])]:
        g=fits[fits.model.isin(models)];selected=g.loc[g.groupby('MouseID').cross_entropy.idxmin(),['MouseID','model']].merge(r,on=['MouseID','model']).set_index('MouseID').reindex(mse.index)
        residual=selected.weighted_mse.to_numpy();boot=1-residual[indices].mean(axis=1)/vm;lo,hi=np.quantile(boot,[.025,.975])
        families.append(dict(family=name,n_animals=len(mse),pooled_within_R2=1-residual.mean()/den,low=lo,high=hi,interpretation='best variant chosen on the same fitted data; optimistic descriptive comparison, not held-out performance'))
    save_table(pd.DataFrame(families),'variance_family_summary')
    cv=extend_cv();cvmean=cv.groupby(['MouseID','model']).heldout_weighted_mse.mean().unstack().reindex(mse.index)
    bc=cvmean.to_numpy()[indices].mean(axis=1);biasboot=bc[:,cvmean.columns.get_loc('bias')]
    cvrows=[]
    for k,model in enumerate(cvmean.columns):
        q=1-bc[:,k]/biasboot;lo,hi=np.quantile(q,[.025,.975]);el,eh=np.quantile(bc[:,k],[.025,.975])
        cvrows.append(dict(model=model,n_animals=len(cvmean),mean_heldout_mse=cvmean[model].mean(),mse_low=el,mse_high=eh,skill_vs_training_bias=1-cvmean[model].mean()/cvmean.bias.mean(),skill_low=lo,skill_high=hi,bootstrap_repeats=repeats))
    save_table(pd.DataFrame(cvrows),'variance_heldout_summary')
    print(pd.DataFrame(summary)[['model','pooled_within_R2','error_reduction_vs_random']].to_string(index=False),flush=True)
    print(pd.DataFrame(cvrows)[['model','mean_heldout_mse','skill_vs_training_bias']].to_string(index=False),flush=True)

