"""Fresh fitting from the single combined dataframe, compared with saved fits.

No saved parameter is used as an initial value. The same deterministic starts,
nested-model starts, bounds and objective as the primary fitter are used.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
import importlib.metadata
import os
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def refit_animal(mid):
    from models import fit_one, predict, allocate, MODEL_NAMES
    bins=pd.read_csv(ROOT/'data/analysis_bins.csv')
    g=bins[bins.modeling_cohort & bins.MouseID.eq(mid)].sort_values('Time_start')
    zt=g.ZT.to_numpy(float);elapsed=g.elapsed_h.to_numpy(float)
    idx=g.bin_index.to_numpy(np.int64);y=g.preference.to_numpy(float)
    weights=g.total_g.to_numpy(float);n=allocate(weights,1000)
    rows=[];predictions=[]
    for model in MODEL_NAMES:
        par,loss,diag=fit_one(model,zt,elapsed,idx,n,y,weights,mid,starts=6)
        rows.append(dict(MouseID=mid,model=model,k=len(par),N_assumed=1000,
            cross_entropy=float(loss),pseudo_AIC_1000=2000*loss+2*len(par),
            parameters=par.tolist(),**diag))
        p=predict(model,par,zt,elapsed,idx,n)
        predictions.extend(dict(MouseID=mid,model=model,bin_index=int(i),
                                predicted=float(v)) for i,v in zip(idx,p))
    return rows,predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--limit',type=int,help='Diagnostic subset; omitted for full verification')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(f'Use a new directory: {out}')
    if out==ROOT or ROOT in out.parents and out.parts[len(ROOT.parts)] in ['data','results','code','tables','figures']:
        raise ValueError('Do not replace reference inputs or outputs')
    out.mkdir(parents=True);(out/'animal_checkpoints').mkdir()
    bins=pd.read_csv(ROOT/'data/analysis_bins.csv')
    ids=sorted(bins.loc[bins.modeling_cohort,'MouseID'].unique())
    if args.limit:ids=ids[:args.limit]
    fitrows=[];predrows=[];start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(refit_animal,mid):mid for mid in ids}
        for number,job in enumerate(as_completed(jobs),1):
            rows,pred=job.result();fitrows.extend(rows);predrows.extend(pred)
            (out/'animal_checkpoints'/f'{number:03d}.json').write_text(json.dumps(dict(MouseID=jobs[job],fits=rows,predictions=pred),default=json_scalar))
            print(f'Fresh refit {number}/{len(ids)} animals; {time.time()-start:.1f} s',flush=True)
    fit=pd.DataFrame(fitrows).sort_values(['MouseID','model'])
    pred=pd.DataFrame(predrows).sort_values(['MouseID','model','bin_index'])
    saved=pd.read_csv(ROOT/'results/corrected_expected_model_fits.csv')
    saved=saved[saved.MouseID.isin(ids)].set_index(['MouseID','model'])
    if set(zip(fit.MouseID,fit.model))!=set(saved.index):
        raise AssertionError('Fresh and saved fit keys differ')
    comparisons=[]
    for row in fit.itertuples():
        old=saved.loc[(row.MouseID,row.model)]
        a=np.asarray(row.parameters);b=np.asarray(json.loads(old.parameters))
        delta=np.abs(a-b)
        comparisons.append(dict(MouseID=row.MouseID,model=row.model,
            max_parameter_difference=float(delta.max()) if len(delta) else 0,
            cross_entropy_difference=float(row.cross_entropy-old.cross_entropy),
            pseudo_AIC_difference=float(row.pseudo_AIC_1000-old.pseudo_AIC_1000)))
    comparison=pd.DataFrame(comparisons)
    reference_pred=pd.read_csv(ROOT/'results/corrected_expected_model_predictions.csv')
    joined=pred.merge(reference_pred[['MouseID','model','bin_index','predicted']],
        on=['MouseID','model','bin_index'],suffixes=('_fresh','_saved'),validate='one_to_one')
    summary=dict(n_animals=len(ids),n_fits=len(fit),n_predictions=len(pred),
        complete_primary_cohort=len(ids)==140,
        max_parameter_difference=float(comparison.max_parameter_difference.max()),
        max_cross_entropy_difference=float(abs(comparison.cross_entropy_difference).max()),
        max_pseudo_AIC_difference=float(abs(comparison.pseudo_AIC_difference).max()),
        max_prediction_difference=float(abs(joined.predicted_fresh-joined.predicted_saved).max()),
        elapsed_seconds=time.time()-start,workers=args.workers,
        seeds='SHA256-derived deterministic labels; six base starts plus freshly fitted nested parents',
        saved_parameters_used_as_initial_values=False,
        scope='Fresh primary fits only; no refitting of cross-validation or phase sensitivity analyses')
    summary['input_sha256']=hashlib.sha256((ROOT/'data/analysis_bins.csv').read_bytes()).hexdigest()
    summary['model_code_sha256']=hashlib.sha256((ROOT/'code/models.py').read_bytes()).hexdigest()
    summary['dependencies']={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','numba']}
    summary['threads']={name:os.environ.get(name) for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}
    summary['parameters_match']=summary['max_parameter_difference']<=1e-7
    summary['predictions_match']=summary['max_prediction_difference']<=1e-10
    summary['objectives_match']=summary['max_cross_entropy_difference']<=1e-10
    summary['passed']=all(summary[k] for k in ['parameters_match','predictions_match','objectives_match'])
    fit['parameters']=fit.parameters.map(json.dumps)
    fit.to_csv(out/'fresh_model_fits.csv',index=False)
    pred.to_csv(out/'fresh_model_predictions.csv',index=False)
    comparison.to_csv(out/'comparison_with_saved.csv',index=False)
    (out/'reproducibility_report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if not summary['passed']:raise SystemExit('Fresh optimization differs from saved parameters; see report. Reference results were not changed.')


if __name__=='__main__':main()
