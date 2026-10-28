"""Optional numerical reruns into a new output directory, never the saved reference."""
from common import *
import argparse, json

def table_exports():
    g=pd.read_csv(OUT/'results/group_cosinor.csv').set_index('group')
    phase=pd.read_csv(OUT/'results/paired_phases.csv');phase=phase[phase.legacy_age_cohort]
    rows=[]
    for (sex,age),d in phase.groupby(['Sex','age_group']):
        row=dict(Sex=sex,age_group=age,n=len(d))
        for col,name in [('total_phase','intake'),('pref_phase','preference')]:
            y=d[col].dropna().to_numpy();mu=mean_phase(y)
            row[name+'_circular_mean_h']=mu;row[name+'_wrapped_SEM_h']=stats.sem(wrap_hours(y-mu))
        rows.append(row)
    save_table(pd.DataFrame(rows),'figure2_phase_means_sem')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['statistics','refit','sensitivity','phases','lag']);a=p.parse_args()
    bins=pd.read_csv(OUT/'data/analysis_bins.csv')
    if a.stage=='statistics':
        from statistics import compute_statistics
        from panel_statistics import compute_panel_statistics
        from bootstrap_phase_revision_20260916 import build
        from validation import fitted_predictive_checks,state_space_forecast
        import variance_revision_20260916 as variance
        compute_statistics(bins);compute_panel_statistics(bins);build()
        fitted_predictive_checks(bins);state_space_forecast(bins)
        # This stage recomputes descriptive summaries from saved matched folds.
        # It does not implicitly run a new optimizer because cache paths changed.
        variance.extend_cv=lambda:pd.read_csv(OUT/'results/variance_extended_blocked_cv.csv')
        variance.calculate();table_exports()
        from phase_lag import main as phase_lag
        phase_lag(OUT,OUT/'results')
        import runpy
        runpy.run_path(str(Path(__file__).with_name('genotype_phase_comparisons.py')),run_name='__main__')
        from export_tables import main as export_tables
        export_tables()
    elif a.stage=='lag':
        from phase_lag import main as phase_lag
        phase_lag(OUT,OUT/'results')
    elif a.stage=='refit':
        from models import model_refits
        from variance_revision_20260916 import extend_cv
        model_refits(bins);extend_cv(force=True)
    elif a.stage=='sensitivity':
        from models import illustrative_simulations,null_controls,recovery_and_sensitivity
        illustrative_simulations();null_controls(bins)
        recovery_and_sensitivity(bins,pd.read_csv(OUT/'results/corrected_expected_model_fits.csv'))
    elif a.stage=='phases':
        from model_phase_analysis import derive,nuisance_profiles,summarize_profiles,refine_selected_phases
        b,t,s=derive();profiles=nuisance_profiles(b,t,s)
        summarize_profiles(profiles,s);refine_selected_phases(b,s,profiles)
    (OUT/'run_scope.json').write_text(json.dumps({'stage':a.stage,'input':'data/analysis_bins.csv','seed':SEED,
       'note':'Numerical rerun. Optimizer differences may change results across dependency versions; saved references are preserved.'},indent=2))

if __name__=='__main__':main()
