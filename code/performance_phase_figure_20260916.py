"""Main Figure 4: model performance, optimizer sensitivity and phase interpretation."""
from common import *

ORDER=['random','bias','simple','simple_no_swap','reward','reward_no_swap','full','full_no_swap']

def build():
    from figures import plt,save,panel,read,LABEL,MODEL_COLOR,GRAY
    ins=read('variance_model_summary').set_index('model');cv=read('variance_heldout_summary').set_index('model')
    s=read('model_phase_selected_animals');s=s[~s.schedule_unresolved]
    ref=read('model_phase_selected_refinement');ref=ref[~ref.schedule_unresolved]
    fig=plt.figure(figsize=(175/25.4,10.3));gs=fig.add_gridspec(3,2,left=.19,right=.96,bottom=.10,top=.94,hspace=.65,wspace=.64,height_ratios=[1.05,1.05,1.1])
    def forest(ax,table,column,low,high,title,letter,limits,percent=False):
        panel(ax,letter,title)
        for i,model in enumerate(ORDER):
            r=table.loc[model];v=r[column]*(100 if percent else 1);lo=r[low]*(100 if percent else 1);hi=r[high]*(100 if percent else 1)
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',ms=3.5,color=MODEL_COLOR[model],capsize=2,lw=1)
        ax.set_yticks(range(len(ORDER)),[LABEL[m] for m in ORDER]);ax.invert_yaxis();ax.set_xlim(*limits);ax.grid(axis='x',alpha=.16);ax.axvline(0,c='.6',lw=.7)
    ax=fig.add_subplot(gs[0,0]);forest(ax,ins,'error_reduction_vs_random','random_reduction_low','random_reduction_high','In-sample fit','A',(-4,103),True);ax.set_xlabel('Error reduction vs Random (%)');ax.set_xticks([0,50,100])
    ax=fig.add_subplot(gs[0,1]);forest(ax,ins,'pooled_within_R2','R2_low','R2_high','Temporal variance captured','B',(-250,110),True);ax.set_xlabel('Within-animal R² (%)');ax.set_xticks([-200,-100,0,100])
    for i,model in enumerate(ORDER):
        v=100*ins.loc[model,'pooled_within_R2'];ax.text(104,i,f'{v:.0f}',ha='right',va='center',fontsize=8,color=MODEL_COLOR[model],bbox=dict(facecolor='white',edgecolor='none',pad=.2))
    ax=fig.add_subplot(gs[1,0]);forest(ax,cv,'mean_heldout_mse','mse_low','mse_high','Held-out prediction','C',(0,float(cv.loc[ORDER,'mse_high'].max())*1.12));ax.set_xlabel('Conditional held-out MSE');ax.ticklabel_format(axis='x',style='plain');ax.set_xticks([0,.05,.10,.15] if cv.loc[ORDER,'mse_high'].max()<.17 else [0,.1,.2,.3])
    ax=fig.add_subplot(gs[1,1]);panel(ax,'D','Phase estimate sensitivity')
    for mask,color,label in [(ref.score_improvement_N1000.le(2),'.65','Small fit change'),(ref.score_improvement_N1000.gt(2),'#b04b36','Score improves >2')]:
        g=ref[mask];ax.scatter(g.saved_nominal_peak_h,g.refined_nominal_peak_h,s=15,color=color,alpha=.8,label=label)
    ax.plot([0,24],[0,24],'--',c='.35',lw=.8);ax.set(xlim=(-.4,24.4),ylim=(-.4,24.4),xticks=[0,12,24],yticks=[0,12,24],xlabel='Saved modulation peak (h)',ylabel='Refined modulation peak (h)')
    ax.legend(frameon=False,loc='upper left',fontsize=8,handlelength=.8,handletextpad=.3)
    ax.text(.97,.02,f'n={len(ref)}',ha='right',va='bottom',transform=ax.transAxes)
    for col,(y,letter,title,ylabel) in enumerate([('predicted_preference_phase_h','E','Predicted preference timing','Predicted preference phase (h)'),('reward_peak_center_h','F','Latent reward timing','Nominal reward peak (h)')]):
        ax=fig.add_subplot(gs[2,col]);panel(ax,letter,title);g=s.dropna(subset=[y,'observed_preference_phase_h'])
        for model,d in g.groupby('model'):
            strong=d.observed_preference_p.lt(.05)
            ax.scatter(d.loc[~strong,'observed_preference_phase_h'],d.loc[~strong,y],s=15,facecolors='none',edgecolors=MODEL_COLOR[model],alpha=.65,lw=.6)
            ax.scatter(d.loc[strong,'observed_preference_phase_h'],d.loc[strong,y],s=22,c=MODEL_COLOR[model],edgecolors='k',lw=.4)
        ax.plot([0,24],[0,24],'--',c='.4',lw=.8);ax.set(xlim=(-.4,24.4),ylim=(-.4,24.4),xticks=[0,6,12,18,24],yticks=[0,6,12,18,24],xlabel='Observed preference phase (h)',ylabel=ylabel)
        error=np.median(abs(wrap_hours(g[y]-g.observed_preference_phase_h)))
        ax.text(.03,.97,f'n={len(g)}; median error {error:.2f} h',va='top',transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    fig.text(.55,.987,'Model fit, prediction and the limits of latent phase interpretation',ha='center',va='top',fontsize=10)
    fig.text(.5,.012,'A–C: 140 animals; bars are animal-bootstrap 95% intervals. NS: no switch. D–F: PA excluded; DD uses prior reference time. E/F use saved fits; phase alignment is descriptive on fitted data.',ha='center',fontsize=8)
    save(fig,'Figure4')

