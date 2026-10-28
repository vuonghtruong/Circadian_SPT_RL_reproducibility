"""Readable phase figures from the frozen primary and sensitivity tables.

No parameters, predictions, cohorts, or model classifications are changed.
The conditional phase-search display uses ell=-1000*cross_entropy rather than
the equivalent minimization score 2000*cross_entropy+2*k.
"""
import json
import numpy as np
import pandas as pd
from common import wrap_hours,mean_phase,resultant,rng_for

ORDER=['random','bias','simple','simple_no_swap','reward','reward_no_swap','full','full_no_swap']
PHASE_MODELS=['reward','reward_no_swap','full','full_no_swap']


def phase_axes(ax,xlabel,ylabel):
    ax.plot([0,24],[0,24],'--',c='.4',lw=1)
    ax.set(xlim=(-.4,24.4),ylim=(-.4,24.4),xticks=[0,12,24],yticks=[0,12,24],xlabel=xlabel,ylabel=ylabel)


def phase_points(ax,g,x,y,pcol,f):
    for model,d in g.groupby('model'):
        strong=d[pcol].lt(.05)
        ax.scatter(d.loc[~strong,x],d.loc[~strong,y],s=15,facecolors='none',edgecolors=f.MODEL_COLOR[model],alpha=.65,lw=.7)
        ax.scatter(d.loc[strong,x],d.loc[strong,y],s=22,c=f.MODEL_COLOR[model],edgecolors='k',lw=.4)


def search_points(ax,ref):
    improved=ref.score_improvement_N1000.gt(2)
    ax.scatter(ref.loc[~improved,'saved_nominal_peak_h'],ref.loc[~improved,'refined_nominal_peak_h'],s=17,c='.6',alpha=.8)
    ax.scatter(ref.loc[improved,'saved_nominal_peak_h'],ref.loc[improved,'refined_nominal_peak_h'],s=24,c='#b04b36',alpha=.9)
    phase_axes(ax,'Initial fit peak (h)','Extended search peak (h)')


def figure4(f):
    ins=f.read('variance_model_summary').set_index('model');cv=f.read('variance_heldout_summary').set_index('model')
    s=f.read('model_phase_selected_animals');s=s[~s.schedule_unresolved]
    ref=f.read('model_phase_selected_refinement');ref=ref[~ref.schedule_unresolved]
    fig=f.plt.figure(figsize=(175/25.4,180/25.4))
    gs=fig.add_gridspec(3,2,left=.18,right=.97,bottom=.085,top=.95,hspace=.78,wspace=.69,height_ratios=[1.05,1.05,1])
    def forest(ax,table,column,low,high,title,letter,limits,percent=False):
        f.panel(ax,letter,title)
        for i,model in enumerate(ORDER):
            r=table.loc[model];mult=100 if percent else 1
            v,lo,hi=r[[column,low,high]].to_numpy(float)*mult
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',ms=3.5,color=f.MODEL_COLOR[model],capsize=2,lw=1.2)
        ax.set_yticks(range(len(ORDER)),[f.LABEL[m] for m in ORDER]);ax.invert_yaxis();ax.set_xlim(*limits)
        ax.grid(axis='x',alpha=.16);ax.axvline(0,c='.6',lw=.7)
    ax=fig.add_subplot(gs[0,0]);forest(ax,ins,'error_reduction_vs_random','random_reduction_low','random_reduction_high','Overall fit','A',(-4,103),True)
    ax.set(xlabel='Error reduction vs Random (%)',xticks=[0,50,100])
    ax=fig.add_subplot(gs[0,1]);forest(ax,ins,'pooled_within_R2','R2_low','R2_high','Circadian variation','B',(-250,110),True)
    ax.set(xlabel='Variation explained (%)',xticks=[-200,-100,0,100])
    ax=fig.add_subplot(gs[1,0]);forest(ax,cv,'mean_heldout_mse','mse_low','mse_high','Withheld measurements','C',(0,float(cv.loc[ORDER,'mse_high'].max())*1.12))
    ax.set_xlabel('Prediction error (MSE)');ax.ticklabel_format(axis='x',style='plain');ax.set_xticks([0,.05,.10,.15] if cv.loc[ORDER,'mse_high'].max()<.17 else [0,.1,.2,.3])
    ax=fig.add_subplot(gs[1,1]);f.panel(ax,'D','Search stability');search_points(ax,ref)
    for col,(y,letter,title,ylabel) in enumerate([
        ('predicted_preference_phase_h','E','Preference timing','Fitted preference peak (h)'),
        ('reward_peak_center_h','F','Reward timing','Model reward peak (h)')]):
        ax=fig.add_subplot(gs[2,col]);f.panel(ax,letter,title)
        g=s.dropna(subset=[y,'observed_preference_phase_h']);phase_points(ax,g,'observed_preference_phase_h',y,'observed_preference_p',f)
        phase_axes(ax,'Measured preference peak (h)',ylabel)
    f.save(fig,'Figure4')


def supplement_phases(bins,f,renderer):
    from model_phase_analysis import modulation
    from models import allocate,predict
    s=f.read('model_phase_selected_animals');s=s[~s.schedule_unresolved].copy()
    profiles=f.read('model_phase_nuisance_profiles')
    ref=f.read('model_phase_selected_refinement');ref=ref[~ref.schedule_unresolved]
    fig=f.plt.figure(figsize=(175/25.4,195/25.4))
    gs=fig.add_gridspec(3,3,left=.10,right=.97,top=.94,bottom=.12,height_ratios=[.85,1,1.1],hspace=.95,wspace=.85)
    def circle(ax,column,title,letter):
        f.panel(ax,letter,title);ax.set_theta_zero_location('N');ax.set_theta_direction(-1)
        ax.set_xticks(np.arange(4)*np.pi/2,['0','6','12','18']);ax.set_yticks([.5,1],['','']);ax.set_ylim(0,1.1);ax.grid(alpha=.3)
        for model,g in s.dropna(subset=[column]).groupby('model'):
            jitter=rng_for(column+model).uniform(-.035,.035,len(g))
            ax.scatter(g[column]*np.pi/12,1+jitter,s=15,color=f.MODEL_COLOR[model],alpha=.7)
        x=s[column].dropna();ax.annotate('',xy=(mean_phase(x)*np.pi/12,resultant(x)),xytext=(0,0),arrowprops={'arrowstyle':'->','lw':2,'color':'k'})
    circle(fig.add_subplot(gs[0,0],projection='polar'),'reward_peak_center_h','Reward peak','A')
    circle(fig.add_subplot(gs[0,1],projection='polar'),'beta_maximum_center_h','Sensitivity peak','B')
    ax=fig.add_subplot(gs[0,2]);ax.axis('off')
    for i,model in enumerate(PHASE_MODELS):
        y=.85-i*.20;ax.scatter(.10,y,s=24,c=f.MODEL_COLOR[model],transform=ax.transAxes)
        ax.text(.25,y,f.LABEL[model],va='center',transform=ax.transAxes)
    for col,(x,y,pcol,letter,title,xlabel,ylabel) in enumerate([
      ('observed_total_phase_h','reward_peak_center_h','observed_total_p','C','Intake timing','Measured intake peak (h)','Model reward peak (h)'),
      ('observed_preference_phase_h','reward_peak_center_h','observed_preference_p','D','Preference timing','Measured preference peak (h)','Model reward peak (h)'),
      ('observed_preference_phase_h','predicted_preference_phase_h','observed_preference_p','E','Fitted timing','Measured preference peak (h)','Fitted preference peak (h)')]):
        ax=fig.add_subplot(gs[1,col]);f.panel(ax,letter,title);g=s.dropna(subset=[x,y]);phase_points(ax,g,x,y,pcol,f);phase_axes(ax,xlabel,ylabel)
    ax=fig.add_subplot(gs[2,0]);f.panel(ax,'F','Model comparison')
    v=f.read('model_phase_variant_stability');v=v[~v.schedule_unresolved]
    variant_counts={}
    for (a,b),g in v.groupby(['model_a','model_b']):
        d=np.sort(abs(g.reward_phase_difference_h.dropna()))
        variant_counts[a+'/'+b]=len(d)
        ax.step(d,np.arange(1,len(d)+1)/len(d),where='post',label=f'{f.LABEL[a]} / '+('NS' if b.endswith('_no_swap') else f.LABEL[b]))
    ax.axhline(.5,c='.6',ls=':');ax.set(xlim=(0,12),ylim=(0,1.02),xticks=[0,6,12],xlabel='Reward peak difference (h)',ylabel='Fraction of animals')
    ax.legend(fontsize=8,loc='lower right',frameon=False,handlelength=.8,handletextpad=.3)
    ax=fig.add_subplot(gs[2,1]);f.panel(ax,'G','Phase search')
    g=s[s.effective_modulation_active].sort_values('nominal_modulation_peak_h');matrix=[]
    for mid in g.MouseID:
        pp=profiles[profiles.MouseID.eq(mid)];minimum=min(pp.cross_entropy.min(),pp.saved_cross_entropy.iloc[0])
        cells=pp[pp.is_grid].sort_values('nominal_peak_h');assert len(cells)==8
        matrix.append(-1000*(cells.cross_entropy.to_numpy()-minimum))
    matrix=np.asarray(matrix)
    profile_n=len(g)
    im=ax.imshow(np.clip(matrix,-5,0),aspect='auto',origin='lower',cmap='Greys',extent=[-1.5,22.5,0,len(g)],vmin=-5,vmax=0)
    ax.scatter((g.nominal_modulation_peak_h.to_numpy()+1.5)%24-1.5,np.arange(len(g))+.5,s=4,c='#c25941')
    ax.set(xticks=[0,6,12,18],xlabel='Tested phase (h)',ylabel='Animal order')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',shrink=1,pad=.30,ticks=[-5,0]);cb.set_label('Δ log-likelihood')
    ax=fig.add_subplot(gs[2,2]);f.panel(ax,'H','Search stability');search_points(ax,ref)
    f.save(fig,'FigureS10_model_phases')
    # The same four frozen examples retain the original quantile selection.
    example=f.read('model_phase_examples')
    candidates=s[s.reward_active].copy();candidates['disagreement']=abs(wrap_hours(candidates.reward_peak_center_h-candidates.predicted_preference_phase_h))
    candidates=candidates.sort_values(['disagreement','MouseID']);indices=np.round(np.array([.125,.375,.625,.875])*(len(candidates)-1)).astype(int)
    assert list(example.MouseID)==list(candidates.iloc[indices].MouseID)
    fig,axs=f.plt.subplots(4,3,figsize=(175/25.4,190/25.4),gridspec_kw={'left':.10,'right':.89,'top':.92,'bottom':.075,'wspace':.75,'hspace':.64})
    examples=[]
    for row,(_,r) in enumerate(example.iterrows()):
        g=bins[bins.MouseID.eq(r.MouseID)].sort_values('ZT');rr=ref[ref.MouseID.eq(r.MouseID)].iloc[0]
        par=np.array(json.loads(rr.parameters));t=np.linspace(0,24,481);reward,beta=modulation(par,t)
        for col,ax in enumerate(axs[row]):
            f.shade(ax);f.panel(ax,chr(65+row*3+col),['Intake','Preference','Model terms'][col] if row==0 else '')
            ax.set_xticks([0,12,24]);ax.set_xlim(0,24)
            if row<3:ax.tick_params(labelbottom=False)
        ax=axs[row,0];ax.plot(g.ZT,g.total_g,'o-',color='#444444',ms=3)
        ax.axvline(r.observed_total_phase_h,ls=':',c='#444444');ax.set_ylabel('Intake (g)')
        chron=bins[bins.MouseID.eq(r.MouseID)].sort_values('Time_start')
        pred=predict(r.model,par,chron.ZT.to_numpy(float),chron.elapsed_h.to_numpy(float),chron.bin_index.to_numpy(np.int64),allocate(chron.total_g.to_numpy(float),1000))
        pp=pd.DataFrame({'ZT':chron.ZT,'fitted':pred}).sort_values('ZT')
        ax=axs[row,1];ax.plot(g.ZT,g.preference,'o',c='#555555',ms=4);ax.plot(pp.ZT,pp.fitted,'s-',c='#b04b36',ms=3)
        ax.axvline(r.observed_preference_phase_h,ls=':',c='#555555');ax.axvline(rr.refined_predicted_phase_h,ls='--',c='#b04b36')
        ax.set(ylim=(-.02,1.02),yticks=[0,.5,1],ylabel='Preference')
        ax=axs[row,2];other=ax.twinx();ax.plot(t,reward,c='#d67a23');other.plot(t,beta,c='#367e8b')
        ax.axvline(rr.refined_reward_peak_h,ls='--',c='#d67a23');ax.set(ylim=(0,1),yticks=[0,.5,1],ylabel='Reward')
        other.set_ylabel('Sensitivity β',color='#367e8b');other.spines['top'].set_visible(False);other.tick_params(axis='y',colors='#367e8b')
        if row<3:other.tick_params(labelbottom=False)
        examples.append({'panels':f'{chr(65+row*3)}–{chr(67+row*3)}','MouseID':r.MouseID,'model':r.model,'condition':r.Condition})
    for ax in axs[-1]:ax.set_xlabel('Reference time (h)')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker='o',color='#555555',linestyle='none'),Line2D([],[],marker='s',color='#b04b36'),Line2D([],[],color='#d67a23'),Line2D([],[],color='#367e8b')]
    fig.legend(handles,['Measured','Fitted','Reward','Sensitivity'],loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False,columnspacing=.9,handlelength=1.3,handletextpad=.4)
    f.save(fig,'FigureS11_model_phase_examples')
    dest=renderer['DEST']/'audit';dest.mkdir(parents=True,exist_ok=True)
    (dest/'phase_clarity_verification.json').write_text(json.dumps({'same_frozen_examples':True,'examples':examples,'phase_search_n':profile_n,'phase_search_transform':'delta_ell=-1000*(CE-min(CE_grid_and_initial)); exactly -old_delta_score/2','phase_search_display_clip':[-5,0],'phase_variant_nonflat_counts':variant_counts,'search_display_n':len(ref),'search_display_log_likelihood_gain_over_1':int(ref.score_improvement_N1000.gt(2).sum()),'search_display_phase_shift_over_3h':int(ref.nominal_phase_shift_h.abs().gt(3).sum()),'no_refitting':True},indent=2))


def install(f,renderer):
    f.figure4=lambda bins:figure4(f)
    renderer['render_supplement_phases']=lambda bins:supplement_phases(bins[bins.modeling_cohort],f,renderer)
    renderer['HEIGHT_MM'].update({'Figure4':180,'FigureS10_model_phases':185,'FigureS11_model_phase_examples':190})
