"""Publication presentation for S4-S10; frozen numerical results are unchanged."""
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


def install(f, renderer):
    renderer['HEIGHT_MM'].update({'FigureS7_reviewer_checks':205,
        'FigureS8_trial_sensitivity':180, 'FigureS9_model_phases':185,
        'FigureS10_model_phase_examples':195})
    order=['random','bias','linear_bias','step_bias','simple','simple_no_swap',
           'reward','reward_no_swap','full','full_no_swap']
    short={'random':'Random','bias':'Bias','linear_bias':'Linear bias','step_bias':'Step bias',
           'simple':'Simple RL','simple_no_swap':'Simple NS','reward':'Reward RL',
           'reward_no_swap':'Reward NS','full':'Full RL','full_no_swap':'Full NS'}
    schedule_colors={'observed':f.BLUE,'uniform':'.50','shuffled':f.ORANGE}

    def model_legend(fig, models, y=.015):
        handles=[Line2D([],[],c=f.MODEL_COLOR[m],marker='o',ms=3,lw=1.3) for m in models]
        fig.legend(handles,[short[m] for m in models],ncol=5,loc='lower center',
            bbox_to_anchor=(.5,y),frameon=False,columnspacing=1.05,handlelength=1.0,
            handletextpad=.35,labelspacing=.45,fontsize=8)

    def axis(ax, letter, title):
        f.panel(ax,letter,title)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True)

    def count_axis(ax, xlabel):
        ax.set_xscale('log');ax.set_xticks([100,1000,10000],['100','1,000','10,000'])
        ax.minorticks_off();ax.set_xlabel(xlabel)

    def checks():
        fig,axs=f.plt.subplots(3,2,figsize=(175/25.4,205/25.4))
        fig.subplots_adjust(left=.12,right=.97,bottom=.17,top=.915,hspace=1.03,wspace=.64)
        ax=axs[0,0];axis(ax,'A','Constant preference')
        prof=f.read('constant_bias_null_profiles');prof=prof[prof.N_assumed.eq(1000)]
        for schedule,color in schedule_colors.items():
            g=prof[prof.schedule.eq(schedule)]
            ax.plot(g.ZT,g['mean'],'o-',c=color,ms=3,label=schedule.capitalize())
            ax.fill_between(g.ZT,g.low,g.high,color=color,alpha=.13,lw=0)
        ax.set(xlabel='Reference time (h)',ylabel='Preference',xticks=[0,12,24],xlim=(0,24))
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.0),
            frameon=False,fontsize=8,handlelength=1.1,columnspacing=1.5)
        ax=axs[0,1];axis(ax,'B','Drinking opportunities')
        c=f.read('trial_availability_counterfactual');families=['bias','simple','reward','full']
        for i,(schedule,color) in enumerate(schedule_colors.items()):
            v=c[c.schedule.eq(schedule)].set_index('model').loc[families,'expected_night_minus_day']
            ax.bar(np.arange(4)+(i-1)*.23,v,.23,color=color,label=schedule.capitalize())
        ax.axhline(0,c='.45',lw=.8)
        ax.set(xticks=range(4),xticklabels=[short[m] for m in families],ylabel='Night − day preference')
        ax.tick_params(axis='x',rotation=35)
        ax=axs[1,0];axis(ax,'C','Cross-validation')
        cv=f.read('expected_model_blocked_cv').groupby(['MouseID','model']).heldout_weighted_mse.mean().unstack()
        means=cv.mean().sort_values()
        ax.bar(range(len(means)),means,yerr=cv.sem().reindex(means.index),
            color=[f.MODEL_COLOR[m] for m in means.index],capsize=2,error_kw={'lw':1.1})
        ax.set(xticks=range(len(means)),xticklabels=[short[m] for m in means.index],ylabel='Prediction error (MSE)')
        ax.tick_params(axis='x',rotation=40)
        ax=axs[1,1];axis(ax,'D','Score-count sensitivity')
        s=f.read('pseudo_count_penalty_sensitivity')
        for m in order:
            g=s[s.model.eq(m)].set_index('pseudo_count')[['n']].reindex([100,1000,10000],fill_value=0).reset_index()
            if len(g):ax.plot(g.pseudo_count,g.n,'o-',c=f.MODEL_COLOR[m],ms=3,lw=1.3)
        count_axis(ax,'Assumed score count');ax.set(ylabel='Best-fit animals',ylim=(0,140),yticks=[0,70,140])
        ax=axs[2,0];axis(ax,'E','Model recovery')
        r=f.read('model_recovery');families=['bias','linear_bias','simple','reward','full']
        ct=f.pd.crosstab(r.true_model,r.selected_model).reindex(index=families,columns=families,fill_value=0)/20
        ax.imshow(ct,vmin=0,vmax=1,cmap='Blues',aspect='equal')
        ax.set(xticks=range(5),xticklabels=[short[m] for m in families],
            yticks=range(5),yticklabels=[short[m] for m in families],xlabel='Selected model',ylabel='Generating model')
        ax.tick_params(axis='x',rotation=40)
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{ct.iloc[i,j]:.0%}',ha='center',va='center',
                color='white' if ct.iloc[i,j]>.5 else '#222222',fontsize=8)
        ax=axs[2,1];axis(ax,'F','Low-intake sensitivity')
        s=f.read('low_intake_sensitivity')
        ax.plot(s.min_total_g,s.n_nominal_sig,'o-',c=f.ORANGE,ms=3,label='Uncorrected')
        ax.plot(s.min_total_g,s.n_FDR_sig,'o-',c=f.BLUE,ms=3,label='BH FDR')
        ax.set(xlabel='Minimum interval intake (g)',ylabel='Significant recordings')
        ax.legend(frameon=False,loc='best',fontsize=8,handlelength=1.0)
        model_legend(fig,order,y=.012)
        f.save(fig,'FigureS7_reviewer_checks')

    def count_sensitivity():
        fig,axs=f.plt.subplots(2,2,figsize=(175/25.4,180/25.4))
        fig.subplots_adjust(left=.12,right=.97,bottom=.19,top=.94,hspace=.72,wspace=.65)
        ax=axs[0,0];axis(ax,'A','Score count only')
        data=f.read('pseudo_count_penalty_sensitivity')
        for m in order:
            g=data[data.model.eq(m)].set_index('pseudo_count')[['n']].reindex([100,1000,10000],fill_value=0).reset_index()
            if len(g):ax.plot(g.pseudo_count,g.n,'o-',c=f.MODEL_COLOR[m],ms=3,lw=1.3)
        count_axis(ax,'Assumed score count');ax.set(ylabel='Best-fit animals',ylim=(0,140),yticks=[0,70,140])
        sample=f.read('latent_count_shape_refit_sensitivity')
        sine=sample[sample['shape'].eq('sine')];ids=sorted(sine.MouseID.unique())
        families=['bias','linear_bias','simple','reward','full']
        winners=sine.loc[sine.groupby(['MouseID','N_assumed']).pseudo_AIC.idxmin()]
        counts=f.pd.crosstab(winners.N_assumed,winners.model).reindex(columns=families,fill_value=0)
        ax=axs[0,1];axis(ax,'B','Event-count refits')
        for m in families:ax.plot(counts.index,counts[m],'o-',c=f.MODEL_COLOR[m],ms=3,lw=1.3)
        count_axis(ax,'Assumed events');ax.set(ylabel='Best-fit animals',ylim=(0,len(ids)+.7),yticks=[0,6,12])
        ax=axs[1,0];axis(ax,'C','Refitted error')
        for m in families:
            g=sine[sine.model.eq(m)].groupby('N_assumed').cross_entropy.agg(['mean','sem'])
            ax.errorbar(g.index,g['mean'],yerr=g['sem'],c=f.MODEL_COLOR[m],fmt='o-',capsize=2,ms=3,lw=1.3)
        count_axis(ax,'Assumed events');ax.set_ylabel('Cross-entropy')
        ax=axs[1,1];axis(ax,'D','Event count and waveform')
        rows=[]
        for shape in ['sine','square']:
            for n in [100,1000,10000]:
                g=sample[sample['shape'].eq(shape)&sample.N_assumed.eq(n)]
                best=g.loc[g.groupby('MouseID').pseudo_AIC.idxmin()].set_index('MouseID').model
                rows.append([families.index(best.loc[mid]) for mid in ids])
        ax.imshow(rows,aspect='auto',interpolation='nearest',
            cmap=ListedColormap([f.MODEL_COLOR[m] for m in families]),vmin=-.5,vmax=4.5)
        ax.set(yticks=range(6),yticklabels=[f'{shape} {n}' for shape in ['Sine','Square'] for n in ['100','1k','10k']],
            xticks=[0,5,11],xticklabels=[1,6,12],xlabel='Animal order')
        f.save_table(f.pd.DataFrame(dict(plot_index=np.arange(1,len(ids)+1),MouseID=ids)),'figureS8_animal_order')
        model_legend(fig,order,y=.025)
        f.save(fig,'FigureS8_trial_sensitivity')

    f.reviewer_supplement=checks
    f.trial_sensitivity_figure=count_sensitivity
    previous_save=f.save

    def save(fig,name):
        if name in ('FigureS10_model_phases','FigureS9_model_phases'):
            # Larger vertical spacing is obtained from the 200-mm page; shorten
            # repeat axis labels without changing any numerical coordinates.
            for ax in fig.axes:
                label=ax.get_xlabel()
                ax.set_xlabel(label.replace('Measured preference peak','Measured peak').replace('Measured intake peak','Measured peak'))
                ax.set_ylabel(ax.get_ylabel().replace('Model reward peak','Reward peak').replace('Fitted preference peak','Fitted preference'))
                leg=ax.get_legend()
                if leg:
                    leg.set_loc('lower right')
                    for t in leg.get_texts():
                        t.set_text(t.get_text().replace('CircRew RL','Reward RL').replace('SimpleRL','Simple RL'))
        if name in ('FigureS11_model_phase_examples','FigureS10_model_phase_examples'):
            # Identify each example on the intake column, leaving the model
            # curves and empirical points unchanged.
            first_axes=[ax for ax in fig.axes if ax.get_ylabel()=='Intake (g)']
            rows=[('R1','Control LD'),('R10','AB LD'),('NR7','AB DD'),('L12','VB · saccharin')]
            for ax,(mouse,condition) in zip(first_axes,rows):
                lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+(hi-lo)*.22)
                ax.text(.03,.96,f'{mouse} · {condition}',transform=ax.transAxes,
                    va='top',ha='left',fontsize=8,color='#333333',
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1.2})
        previous_save(fig,name)
    f.save=save
