"""Derive model-implied phases; assess behavioral alignment and phase sensitivity.

This analysis preserves primary fits and cohorts. Latent phases are descriptive
parameters, not directly measured reward phases or independent validation.
"""
from common import *
from models import predict,allocate,fit_one,bounds_for
from statistics import cosinor
import json,time,argparse,inspect
from scipy.optimize import minimize

PHASE_MODELS=['reward','reward_no_swap','full','full_no_swap']
PHASE_TOL=1e-5

def read(name):return pd.read_csv(OUT/'results'/f'{name}.csv')

def phase_to_peak(phi):return (6-12*np.asarray(phi)/np.pi)%24

def modulation(par,t):
    c=np.sin(np.asarray(t)*np.pi/12+par[4])
    r=np.clip(par[2]+par[3]*c,.05,.95)
    beta=np.maximum(.2,np.exp(par[1])*(1-(par[5] if len(par)>5 else 0)*c))
    return r,beta

def loss_for(g,model,par,N=1000):
    valid=g.preference.notna().to_numpy()&(g.total_g.to_numpy()>0)
    y=g.preference.to_numpy(float)[valid];w=g.total_g.to_numpy(float)[valid];w=w/w.sum()
    p=predict(model,np.asarray(par),g.ZT.to_numpy(float),g.elapsed_h.to_numpy(float),g.bin_index.to_numpy(np.int64),allocate(g.total_g.to_numpy(float),N))
    p=np.clip(p[valid],1e-10,1-1e-10)
    return float(-np.sum(w*(y*np.log(p)+(1-y)*np.log1p(-p)))),p

def derive():
    bins=pd.read_csv(OUT/'data/analysis_bins.csv');bins=bins[bins.modeling_cohort]
    fits=read('corrected_expected_model_fits');fits=fits[fits.model.isin(PHASE_MODELS)]
    cls=read('model_selection').set_index('MouseID').model
    phases=read('paired_phases').set_index('MouseID');audit=read('light_dark_mouse_audit').set_index('MouseID')
    pred=read('corrected_expected_model_predictions');rows=[];grid=np.arange(0,24,.01)
    for _,row in fits.iterrows():
        g=bins[bins.MouseID.eq(row.MouseID)].sort_values('Time_start');meta=g.iloc[0];par=np.array(json.loads(row.parameters));peak=float(phase_to_peak(par[4]));r,b=modulation(par,grid)
        pp=pred[(pred.MouseID==row.MouseID)&(pred.model==row.model)];pc=cosinor(pp.ZT,pp.predicted)
        rr=phases.loc[row.MouseID];aa=audit.loc[row.MouseID]
        active_reward=np.ptp(r)>PHASE_TOL;active_beta=np.ptp(b)>PHASE_TOL
        rows.append(dict(MouseID=row.MouseID,model=row.model,selected_eight_model=row.model==cls[row.MouseID],
            Sex=meta.Sex,Age=meta.Age,Genotype=meta.Genotype,Condition=meta.Condition,Condition_real=meta.Condition_real,
            inferred_animal_id=meta.inferred_animal_id,fluid=meta.fluid,schedule=aa.schedule,schedule_unresolved=aa.source_timing_conflict,
            day_dominant=aa.day_dominant,day_fraction=aa.day_fraction,
            phi_radians=par[4],nominal_modulation_peak_h=peak,
            reward_peak_center_h=peak if active_reward else np.nan,
            beta_minimum_center_h=peak if active_beta else np.nan,beta_maximum_center_h=(peak+12)%24 if active_beta else np.nan,
            reward_amplitude=par[3],beta_modulation_amplitude=par[5] if len(par)>5 else 0.,
            reward_effective_range=np.ptp(r),beta_effective_range=np.ptp(b),reward_active=active_reward,beta_active=active_beta,
            reward_maximum_plateau_h=float(np.sum(r>=.95-1e-10)*.01) if active_reward else 24.,
            beta_minimum_floor_plateau_h=float(np.sum(b<=.2+1e-10)*.01) if active_beta else 24.,
            effective_modulation_active=active_reward or active_beta,
            predicted_preference_phase_h=pc['phase_h'] if pc['amplitude']>1e-5 else np.nan,
            predicted_preference_amplitude=pc['amplitude'],predicted_preference_r2=pc['r2'],
            observed_total_phase_h=rr.total_phase,observed_preference_phase_h=rr.pref_phase,
            observed_total_p=rr.total_p,observed_preference_p=rr.pref_p,
            observed_total_q=rr.total_q,observed_preference_q=rr.pref_q,
            reward_minus_total_h=float(wrap_hours(peak-rr.total_phase)) if active_reward else np.nan,
            reward_minus_preference_h=float(wrap_hours(peak-rr.pref_phase)) if active_reward else np.nan,
            predicted_minus_observed_preference_h=float(wrap_hours(pc['phase_h']-rr.pref_phase)),
            cross_entropy=row.cross_entropy,converged=row.converged,parameters=row.parameters))
    table=pd.DataFrame(rows);save_table(table,'model_phase_parameters')
    selected=table[table.selected_eight_model].copy();save_table(selected,'model_phase_selected_animals')
    summaries=[]
    cohorts={'all selected phase models':selected,'exclude unresolved PA':selected[~selected.schedule_unresolved],
        'exclude day-dominant':selected[~selected.day_dominant],
        'standard LD/reference only':selected[selected.schedule.eq('standard LD/reference')],
        'DD reference only':selected[selected.schedule.eq('DD')]}
    for cohort,g in cohorts.items():
        for metric in ['reward_peak_center_h','beta_maximum_center_h','predicted_preference_phase_h','observed_total_phase_h','observed_preference_phase_h']:
            x=g[metric].dropna().to_numpy();mu=mean_phase(x)
            summaries.append(dict(cohort=cohort,metric=metric,n=len(x),mean_phase_h=mu,resultant_R=resultant(x) if len(x) else np.nan))
    save_table(pd.DataFrame(summaries),'model_phase_population_summary')
    align=[]
    relationships=[('reward_peak_center_h','observed_total_phase_h','observed_total_p'),
        ('reward_peak_center_h','observed_preference_phase_h','observed_preference_p'),
        ('predicted_preference_phase_h','observed_preference_phase_h','observed_preference_p')]
    for cohort,g in cohorts.items():
        for left,right,pcol in relationships:
            for subset in ['all estimable','observed rhythm p<0.05']:
                s=g if subset=='all estimable' else g[g[pcol]<.05]
                s=s.dropna(subset=[left,right]);d=wrap_hours(s[left].to_numpy()-s[right].to_numpy())
                if len(d)==0:continue
                mu=float(wrap_hours(mean_phase(d)));rg=rng_for(cohort+left+subset)
                boot=np.array([float(wrap_hours(mean_phase(rg.choice(d,len(d),replace=True)))) for _ in range(2000)])
                offsets=wrap_hours(boot-mu);lo,hi=np.quantile(offsets,[.025,.975])
                align.append(dict(cohort=cohort,left=left,right=right,subset=subset,n=len(d),
                    circular_mean_difference_h=mu,mean_difference_ci_lower_h=mu+lo,mean_difference_ci_upper_h=mu+hi,
                    lag_resultant_R=resultant(d),median_absolute_difference_h=np.median(abs(d)),within_3h_fraction=np.mean(abs(d)<=3)))
    save_table(pd.DataFrame(align),'model_phase_behavior_alignment')
    # Model-structure robustness compares the same animals, not separate cohorts.
    variants=[]
    for a,b in [('full','reward'),('full','full_no_swap'),('reward','reward_no_swap')]:
        x=table[table.model.eq(a)].set_index('MouseID');y=table[table.model.eq(b)].set_index('MouseID')
        for mid in x.index:
            delta=float(wrap_hours(x.loc[mid,'reward_peak_center_h']-y.loc[mid,'reward_peak_center_h']))
            variants.append(dict(MouseID=mid,model_a=a,model_b=b,reward_phase_difference_h=delta,schedule_unresolved=bool(x.loc[mid,'schedule_unresolved'])))
    save_table(pd.DataFrame(variants),'model_phase_variant_stability')
    # Isolated phase slices hold all nuisance parameters fixed. No CI claim.
    slices=[];widths=[]
    for _,row in selected.iterrows():
        if not row.effective_modulation_active:continue
        g=bins[bins.MouseID==row.MouseID].sort_values('Time_start');par=np.array(json.loads(row.parameters));values=[]
        for offset in np.arange(-12,12,.25):
            q=par.copy();q[4]=(par[4]-offset*np.pi/12)%(2*np.pi);loss,_=loss_for(g,row.model,q)
            values.append((offset,loss));slices.append(dict(MouseID=row.MouseID,model=row.model,peak_offset_h=offset,cross_entropy=loss,delta_CE_from_saved=loss-row.cross_entropy))
        v=np.array(values);minimum=min(row.cross_entropy,v[:,1].min())
        for count in [100,1000,10000]:
            near=2*count*(v[:,1]-minimum)<=2+1e-10
            widths.append(dict(MouseID=row.MouseID,model=row.model,N_score=count,conditional_near_optimal_hours=float(near.sum()*.25),
                max_loss_improvement_from_slice=max(0,row.cross_entropy-minimum)))
    save_table(pd.DataFrame(slices),'model_phase_conditional_slices');save_table(pd.DataFrame(widths),'model_phase_conditional_widths')
    print(f'Extracted {len(table)} phase-model fits; {len(selected)} phase-model winners. Active reward={selected.reward_active.sum()}, active beta={selected.beta_active.sum()}',flush=True)
    print(pd.DataFrame(align).query("cohort == 'exclude unresolved PA' and subset == 'all estimable'").to_string(index=False),flush=True)
    return bins,table,selected

def nuisance_profiles(bins,table,selected):
    """Fixed phase, optimize nuisance parameters on every selected phase model.

    Coarse 3-h grid plus saved phase, two feasible starts; objective-support
    intervals only, never confidence intervals from an assumed count likelihood.
    """
    rows=[];t0=time.time();allfits=read('corrected_expected_model_fits').set_index(['MouseID','model'])
    for j,(_,row) in enumerate(selected.iterrows()):
        if not row.effective_modulation_active:continue
        g=bins[bins.MouseID==row.MouseID].sort_values('Time_start');saved=np.array(json.loads(row.parameters));free=[i for i in range(len(saved)) if i!=4];bounds=np.array(bounds_for(row.model))[free]
        zt=g.ZT.to_numpy(float);elapsed=g.elapsed_h.to_numpy(float);idx=g.bin_index.to_numpy(np.int64);n=allocate(g.total_g.to_numpy(float),1000)
        valid=g.preference.notna().to_numpy()&(g.total_g.to_numpy()>0);y=g.preference.to_numpy(float)[valid];weight=g.total_g.to_numpy(float)[valid];weight/=weight.sum()
        parent='reward' if row.model.startswith('full') else 'simple'
        if row.model.endswith('no_swap'):parent+='_no_swap'
        p=np.array(json.loads(allfits.loc[(row.MouseID,parent),'parameters']))
        phases=np.r_[np.arange(0,24,3.),row.nominal_modulation_peak_h]
        for phase_index,phase_h in enumerate(phases):
            phi=((6-phase_h)*np.pi/12)%(2*np.pi)
            def objective(x):
                q=saved.copy();q[free]=x;q[4]=phi
                pred=np.clip(predict(row.model,q,zt,elapsed,idx,n)[valid],1e-10,1-1e-10)
                return -np.sum(weight*(y*np.log(pred)+(1-y)*np.log1p(-pred)))
            # Start at the saved nuisance parameters and an embedded simpler fit.
            parent_full=np.r_[p,0.] if row.model.startswith('full') else np.r_[p,0.,phi]
            seeds=[saved[free],parent_full[free]];best_loss=np.inf;best=None;success=False
            for seed in seeds:
                initial=objective(seed)
                result=minimize(objective,seed,method='L-BFGS-B',bounds=bounds,options={'maxiter':250,'ftol':1e-10,'gtol':1e-6,'maxls':40})
                val=min(initial,result.fun);x=seed if initial<result.fun else result.x
                if val<best_loss:best_loss=val;best=x;success=bool(result.success)
            q=saved.copy();q[free]=best;q[4]=phi
            rows.append(dict(MouseID=row.MouseID,model=row.model,nominal_peak_h=phase_h,cross_entropy=best_loss,
                saved_cross_entropy=row.cross_entropy,converged=success,parameters=json.dumps(q.tolist()),is_grid=phase_index<8))
        if (j+1)%10==0:
            save_table(pd.DataFrame(rows),'model_phase_nuisance_profiles')
            print(f'Phase nuisance profiles: {j+1}/{len(selected)}, {time.time()-t0:.0f}s',flush=True)
    profiles=pd.DataFrame(rows);save_table(profiles,'model_phase_nuisance_profiles')
    digest={'inputs':phase_signature(),'rows':len(profiles),'table_hash':hashlib.sha256((OUT/'results/model_phase_nuisance_profiles.csv').read_bytes()).hexdigest()}
    (OUT/'results/model_phase_profile_cache.json').write_text(json.dumps(digest,indent=2))
    return profiles

def phase_signature():
    paths=[OUT/'data/analysis_bins.csv',OUT/'results/corrected_expected_model_fits.csv',OUT/'results/model_selection.csv',OUT/'results/model_phase_selected_animals.csv',Path(__file__).parent/'models.py']
    result={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result['nuisance_profiles']=hashlib.sha256(inspect.getsource(nuisance_profiles).encode()).hexdigest()
    return result

def summarize_profiles(profiles,selected):
    rows=[]
    for mid,g in profiles.groupby('MouseID'):
        saved=g.saved_cross_entropy.iloc[0];best=min(saved,g.cross_entropy.min());bestrow=g.loc[g.cross_entropy.idxmin()]
        baseline=selected[selected.MouseID.eq(mid)].iloc[0]
        for N in [100,1000,10000]:
            grid=g[g.is_grid];near=2*N*(grid.cross_entropy-best)<=2+1e-8
            # A 3-hour grid yields a coarse support measure, not precise endpoints.
            acceptable=g[2*N*(g.cross_entropy-best)<=2+1e-8]
            shift=abs(wrap_hours(acceptable.nominal_peak_h.to_numpy()-baseline.nominal_modulation_peak_h))
            rows.append(dict(MouseID=mid,model=g.model.iloc[0],N_score=N,n_grid_points_within_2_score=int(near.sum()),
                coarse_near_optimal_hours=int(near.sum())*3,
                has_near_optimal_phase_over_6h_from_saved=bool(np.any(shift>=6-1e-8)),
                best_tested_nominal_peak_h=bestrow.nominal_peak_h,
                best_tested_peak_shift_h=float(wrap_hours(bestrow.nominal_peak_h-baseline.nominal_modulation_peak_h)),
                best_CE=best,saved_CE=saved,improvement_in_score=2*N*(saved-best),
                saved_fit_outside_2_score=2*N*(saved-best)>2+1e-8,
                schedule_unresolved=baseline.schedule_unresolved))
    out=pd.DataFrame(rows);save_table(out,'model_phase_profile_support')
    print(out[out.N_score.eq(1000)].agg({'coarse_near_optimal_hours':'median','has_near_optimal_phase_over_6h_from_saved':'sum','saved_fit_outside_2_score':'sum','improvement_in_score':'max'}).to_dict(),flush=True)
    return out

def refine_selected_phases(bins,selected,profiles):
    """Phase-focused sensitivity in each already-selected family, not reranking."""
    rows=[];start=time.time()
    for j,(_,r) in enumerate(selected.iterrows()):
        g=bins[bins.MouseID.eq(r.MouseID)].sort_values('Time_start');saved=np.array(json.loads(r.parameters))
        zt=g.ZT.to_numpy(float);elapsed=g.elapsed_h.to_numpy(float);idx=g.bin_index.to_numpy(np.int64);n=allocate(g.total_g.to_numpy(float),1000)
        valid=g.preference.notna().to_numpy()&(g.total_g.to_numpy()>0);y=g.preference.to_numpy(float)[valid];w=g.total_g.to_numpy(float)[valid];w/=w.sum()
        def objective(q):
            p=np.clip(predict(r.model,q,zt,elapsed,idx,n)[valid],1e-10,1-1e-10)
            return -np.sum(w*(y*np.log(p)+(1-y)*np.log1p(-p)))
        candidates=profiles[profiles.MouseID.eq(r.MouseID)].sort_values('cross_entropy').head(3)
        seeds=[saved]+[np.array(json.loads(x)) for x in candidates.parameters]
        for delta in [-np.pi/12,np.pi/12]:
            q=seeds[1].copy();q[4]=(q[4]+delta)%(2*np.pi);seeds.append(q)
        best=saved.copy();bestloss=objective(best);success=bool(r.converged)
        for seed in seeds:
            result=minimize(objective,seed,method='L-BFGS-B',bounds=bounds_for(r.model),options={'maxiter':500,'ftol':1e-11,'gtol':1e-6,'maxls':40})
            for q,ok in [(seed,False),(result.x,bool(result.success))]:
                value=objective(q)
                if value<bestloss-1e-12:best=q.copy();bestloss=value;success=ok
        reward,beta=modulation(best,np.arange(0,24,.01));peak=float(phase_to_peak(best[4]))
        pr=predict(r.model,best,zt,elapsed,idx,n);pc=cosinor(zt,pr)
        rows.append(dict(MouseID=r.MouseID,model=r.model,parameters=json.dumps(best.tolist()),saved_CE=r.cross_entropy,refined_CE=bestloss,
            score_improvement_N1000=2000*(r.cross_entropy-bestloss),converged=success,
            saved_nominal_peak_h=r.nominal_modulation_peak_h,refined_nominal_peak_h=peak,
            nominal_phase_shift_h=float(wrap_hours(peak-r.nominal_modulation_peak_h)),
            refined_reward_peak_h=peak if np.ptp(reward)>PHASE_TOL else np.nan,
            refined_beta_maximum_h=(peak+12)%24 if np.ptp(beta)>PHASE_TOL else np.nan,
            refined_predicted_phase_h=pc['phase_h'] if pc['amplitude']>1e-5 else np.nan,
            refined_predicted_amplitude=pc['amplitude'],refined_predicted_r2=pc['r2'],
            schedule_unresolved=r.schedule_unresolved))
        if (j+1)%25==0:print(f'Phase-focused refinement {j+1}/{len(selected)} ({time.time()-start:.0f}s)',flush=True)
    refined=pd.DataFrame(rows);save_table(refined,'model_phase_selected_refinement')
    joined=selected.merge(refined.drop(columns=['parameters','converged','schedule_unresolved']),on=['MouseID','model'],validate='one_to_one')
    out=[]
    for cohort,g in [('all selected phase models',joined),('exclude unresolved PA',joined[~joined.schedule_unresolved]),('exclude day-dominant',joined[~joined.day_dominant])]:
        for version,l,r in [('saved reward','reward_peak_center_h','observed_preference_phase_h'),('refined reward','refined_reward_peak_h','observed_preference_phase_h'),('saved reward vs intake','reward_peak_center_h','observed_total_phase_h'),('refined reward vs intake','refined_reward_peak_h','observed_total_phase_h'),('saved prediction','predicted_preference_phase_h','observed_preference_phase_h'),('refined prediction','refined_predicted_phase_h','observed_preference_phase_h')]:
            valid=g.dropna(subset=[l,r]);d=abs(wrap_hours(valid[l]-valid[r]));out.append(dict(cohort=cohort,version=version,n=len(valid),phase_concentration_R=resultant(valid[l]),median_absolute_difference_h=np.median(d),within_3h_fraction=np.mean(d<=3)))
    save_table(pd.DataFrame(out),'model_phase_refinement_behavior')
    print('Refinement:',dict(improved_over_2_score=int((refined.score_improvement_N1000>2).sum()),phase_shift_over_3h=int((abs(refined.nominal_phase_shift_h)>3).sum()),max_score_improvement=refined.score_improvement_N1000.max()),flush=True)
    return refined

def make_figures(bins,table,selected,profiles,support):
    from figures import plt,save,LABEL,MODEL_COLOR,shade
    from matplotlib.colors import Normalize
    s=selected[~selected.schedule_unresolved].copy()
    fig=plt.figure(figsize=(13.5,13));gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.2],hspace=.60,wspace=.45)
    def circle(ax,column,title):
        ax.set_theta_zero_location('N');ax.set_theta_direction(-1);ax.set_xticks(np.arange(4)*np.pi/2,['0','6','12','18']);ax.set_yticks([.5,1],['','']);ax.set_ylim(0,1.1);ax.grid(alpha=.3)
        for model,g in s.dropna(subset=[column]).groupby('model'):
            jitter=rng_for(column+model).uniform(-.035,.035,len(g))
            ax.scatter(g[column]*np.pi/12,1+jitter,s=15,color=MODEL_COLOR[model],alpha=.7,label=f'{LABEL[model]} ({len(g)})')
        x=s[column].dropna();mu=mean_phase(x);R=resultant(x)
        ax.annotate('',xy=(mu*np.pi/12,R),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':2,'color':'k'})
        ax.set_title(title+f'\nn={len(x)}; concentration R={R:.2f}',pad=18)
    circle(fig.add_subplot(gs[0,0],projection='polar'),'reward_peak_center_h','A  Nominal reward-modulation peak')
    circle(fig.add_subplot(gs[0,1],projection='polar'),'beta_maximum_center_h','B  Maximum choice sensitivity β')
    ax=fig.add_subplot(gs[0,2]);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(0,1,'How to read the phases',fontweight='bold',fontsize=11,va='top')
    ax.text(0,.88,'Reward peak center:\n(6 − 12φ/π) mod 24 h\n\nβ maximum: +12 h\n(imposed by the model).\n\nLatent, not measured.\nClipping can flatten peaks.\n\nPA omitted.\nDD: prior reference time.',va='top',fontsize=8,linespacing=1.1)
    for i,model in enumerate(PHASE_MODELS):
        x=.03+(i%2)*.5;y=.13-(i//2)*.09
        ax.scatter(x,y,s=24,c=MODEL_COLOR[model],transform=ax.transAxes);ax.text(x+.07,y,LABEL[model],va='center',transform=ax.transAxes,fontsize=8)
    def paired(ax,x,y,pcol,title):
        g=s.dropna(subset=[x,y]);delta=abs(wrap_hours(g[y]-g[x]))
        for model,d in g.groupby('model'):
            weak=d[d[pcol]>=.05];strong=d[d[pcol]<.05]
            ax.scatter(weak[x],weak[y],s=19,facecolor='none',edgecolor=MODEL_COLOR[model],alpha=.6,lw=.7)
            ax.scatter(strong[x],strong[y],s=28,c=MODEL_COLOR[model],edgecolor='k',lw=.35)
        ax.plot([0,24],[0,24],'k--',lw=1);ax.axvline(12,c='.7',lw=.7);ax.axhline(12,c='.7',lw=.7)
        ax.set(xlim=(0,24),ylim=(0,24),xticks=[0,6,12,18,24],yticks=[0,6,12,18,24],title=title)
        ax.text(.03,.98,f'n={len(g)}\nmedian |Δphase|={np.median(delta):.2f} h',transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    ax=fig.add_subplot(gs[1,0]);paired(ax,'observed_total_phase_h','reward_peak_center_h','observed_total_p','C  Reward parameter vs intake')
    ax.set(xlabel='Observed intake cosinor phase (h)',ylabel='Nominal reward peak (h)')
    ax=fig.add_subplot(gs[1,1]);paired(ax,'observed_preference_phase_h','reward_peak_center_h','observed_preference_p','D  Reward parameter vs preference')
    ax.set(xlabel='Observed preference cosinor phase (h)',ylabel='Nominal reward peak (h)')
    ax=fig.add_subplot(gs[1,2]);paired(ax,'observed_preference_phase_h','predicted_preference_phase_h','observed_preference_p','E  Predicted vs observed preference')
    ax.set(xlabel='Observed preference cosinor phase (h)',ylabel='Predicted preference cosinor phase (h)')
    ax=fig.add_subplot(gs[2,0]);v=read('model_phase_variant_stability');v=v[~v.schedule_unresolved]
    for (a,b),g in v.groupby(['model_a','model_b']):
        d=np.sort(abs(g.reward_phase_difference_h.dropna()));ax.step(d,np.arange(1,len(d)+1)/len(d),where='post',label=f'{LABEL[a]}/{LABEL[b]} ({len(d)})')
    ax.axhline(.5,c='.6',ls=':');ax.set(xlim=(0,12),ylim=(0,1.02),xlabel='Absolute reward-phase difference (h)',ylabel='Cumulative fraction of animals',title='F  Phase changes across model structures');ax.legend(fontsize=7,loc='lower right')
    ax=fig.add_subplot(gs[2,1]);g=s[s.effective_modulation_active].sort_values('nominal_modulation_peak_h');matrix=[]
    for mid in g.MouseID:
        pp=profiles[profiles.MouseID.eq(mid)];minimum=min(pp.cross_entropy.min(),pp.saved_cross_entropy.iloc[0]);cells=pp[pp.is_grid].sort_values('nominal_peak_h');assert len(cells)==8
        matrix.append(2000*(cells.cross_entropy.to_numpy()-minimum))
    im=ax.imshow(np.clip(matrix,0,10),aspect='auto',origin='lower',cmap='Greys',extent=[-1.5,22.5,0,len(g)],vmin=0,vmax=10)
    ax.scatter((g.nominal_modulation_peak_h.to_numpy()+1.5)%24-1.5,np.arange(len(g))+.5,s=4,c='#c25941',label='Saved phase')
    ax.set(xticks=[0,6,12,18],xlabel='Fixed modulation phase center (h)',ylabel='Animal order (saved phase)',title=f'G  Nuisance-optimized phase profiles (n={len(g)})')
    fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.85,pad=.25,label='Δ pseudo-score (N=1,000)');ax.legend(fontsize=8,loc='upper left')
    save_table(g[['MouseID','model','nominal_modulation_peak_h']].reset_index(drop=True).rename_axis('plot_order').reset_index(),'model_phase_animal_order')
    ax=fig.add_subplot(gs[2,2]);refined=read('model_phase_selected_refinement');q=refined[~refined.schedule_unresolved]
    improved=q.score_improvement_N1000>2
    ax.scatter(q.loc[~improved,'saved_nominal_peak_h'],q.loc[~improved,'refined_nominal_peak_h'],s=20,c='.5',alpha=.7,label='Small fit change')
    ax.scatter(q.loc[improved,'saved_nominal_peak_h'],q.loc[improved,'refined_nominal_peak_h'],s=25,c='#b04b36',label='Score improves >2')
    ax.plot([0,24],[0,24],'k--',lw=1);ax.set(xlim=(0,24),ylim=(0,24),xticks=[0,6,12,18,24],yticks=[0,6,12,18,24],xlabel='Saved modulation phase center (h)',ylabel='Phase after additional optimization (h)',title='H  Phase-focused refinement in the same family');ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Saved model phases and checks of their behavioral interpretation',fontsize=14,y=.995)
    fig.text(.5,.01,'Clock values use the source reference. Open points: observed rhythm p ≥ .05; filled: nominal p < .05. All comparisons are descriptive and use fitted data.',ha='center',fontsize=8)
    save(fig,'FigureS10_model_phases')
    # Examples cover four quantiles of reward/output phase disagreement, not only
    # especially good fits or especially aligned phases.
    candidates=s[s.reward_active].copy();candidates['disagreement']=abs(wrap_hours(candidates.reward_peak_center_h-candidates.predicted_preference_phase_h));candidates=candidates.sort_values(['disagreement','MouseID'])
    example=candidates.iloc[np.round(np.array([.125,.375,.625,.875])*(len(candidates)-1)).astype(int)].copy();save_table(example,'model_phase_examples')
    fig,axs=plt.subplots(4,3,figsize=(13,11),gridspec_kw={'wspace':.46,'hspace':.85})
    for row,(_,r) in enumerate(example.iterrows()):
        g=bins[bins.MouseID.eq(r.MouseID)].sort_values('ZT');ref=refined[refined.MouseID.eq(r.MouseID)].iloc[0];par=np.array(json.loads(ref.parameters));t=np.linspace(0,24,481);reward,beta=modulation(par,t)
        for ax in axs[row]:shade(ax)
        date=str(g.File.iloc[0])[:8];mouse=g.Mouse.iloc[0]
        ax=axs[row,0];ax.plot(g.ZT,g.total_g,'o-',color='#444444',label='Total intake',ms=3);ax.axvline(r.observed_total_phase_h,ls=':',c='#444444');ax.set(ylabel='Intake (g / 3 h)',title=f'{chr(65+row*3)}  {date} {mouse} | {r.Condition}')
        ax=axs[row,1];chron=bins[bins.MouseID.eq(r.MouseID)].sort_values('Time_start');predq=predict(r.model,par,chron.ZT.to_numpy(float),chron.elapsed_h.to_numpy(float),chron.bin_index.to_numpy(np.int64),allocate(chron.total_g.to_numpy(float),1000));pp=pd.DataFrame({'ZT':chron.ZT,'predicted':predq}).sort_values('ZT')
        ax.plot(g.ZT,g.preference,'o',c='#555555',label='Observed',ms=4);ax.plot(pp.ZT,pp.predicted,'s-',c='#b04b36',label=LABEL[r.model]+' prediction',ms=3)
        if row==0:shared_handles=list(ax.get_lines()[:2])
        ax.axvline(r.observed_preference_phase_h,ls=':',c='#555555');ax.axvline(ref.refined_predicted_phase_h,ls='--',c='#b04b36');ax.set(ylim=(-.02,1.02),ylabel='Preference',title=f'{chr(66+row*3)}  Observed and predicted allocation\n{LABEL[r.model]}')
        ax=axs[row,2];other=ax.twinx();ax.plot(t,reward,c='#d67a23',label='Reward term');other.plot(t,beta,c='#367e8b',label='Sensitivity β');ax.axvline(ref.refined_reward_peak_h,ls='--',c='#d67a23');ax.set(ylim=(0,1),ylabel='Model reward term',title=f'{chr(67+row*3)}  Latent modulation functions');other.set_ylabel('Choice sensitivity β',color='#367e8b');other.spines['top'].set_visible(False)
        if row==0:shared_handles+=list(ax.get_lines()[:1])+list(other.get_lines()[:1])
    fig.suptitle('Examples: predicted preference phase differs from latent reward phase',fontsize=14,y=.99)
    fig.legend(shared_handles,['Observed','Model prediction','Reward term','Sensitivity β'],loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False,columnspacing=1.,handlelength=1.5)
    for ax in axs[-1]:ax.set_xlabel('Source ZT / reference time (h)')
    fig.text(.5,.02,'Phase-focused refined fits within the original selected families. Vertical lines: phase centers. Shading: ZT12–24 (prior reference for DD).',ha='center',fontsize=8)
    fig.text(.5,.005,'Smooth latent curves illustrate model functions; fitting evaluates modulation at the eight bin centers.',ha='center',fontsize=8)
    save(fig,'FigureS11_model_phase_examples')




