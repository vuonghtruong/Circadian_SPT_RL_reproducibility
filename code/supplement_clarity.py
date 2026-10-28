"""Compact S1–S3 using the submitted figure's graphical grammar.

Presentation only: all cohorts, fitted values and tests come from frozen tables.
"""
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def install(f, renderer):
    renderer['HEIGHT_MM'].update({'FigureS1':135,'FigureS2':145,'FigureS3':145})

    def supplement1(bins):
        fig,axs=f.plt.subplots(2,2,figsize=(175/25.4,135/25.4))
        fig.subplots_adjust(left=.105,right=.905,bottom=.105,top=.91,hspace=.78,wspace=.70)
        groups=[('AB',bins[bins.Genotype.eq('AB')]),('VB',bins[bins.Genotype.eq('VB')]),
                ('Glucose',bins[bins.fluid.eq('glucose')]),('Water',bins[bins.fluid.eq('water')])]
        for ax,letter,(title,g) in zip(axs.flat,'ABCD',groups):
            f.panel(ax,letter,title)
            f.individual_lines(ax,g,'total_g',f.BLUE,.15)
            f.profile(ax,g,'total_g',f.BLUE,fit=True)
            ax.set(xlim=(0,24),ylim=(0,5.5),xticks=[0,6,12,18,24],yticks=[0,2,4],xlabel='ZT/CT (h)',ylabel='Total intake (g)')
            ax.tick_params(axis='y',colors=f.BLUE);ax.yaxis.label.set_color(f.BLUE)
            twin=ax.twinx()
            f.individual_lines(twin,g,'preference',f.ORANGE,.15)
            f.profile(twin,g,'preference',f.ORANGE,fit=True)
            twin.set(ylim=(0,1),yticks=[0,.5,1],ylabel='Preference',xlabel='')
            twin.tick_params(axis='y',colors=f.ORANGE);twin.yaxis.label.set_color(f.ORANGE)
            twin.spines['right'].set_visible(True)
        f.save(fig,'FigureS1')

    def supplement2(bins):
        fig,axs=f.plt.subplots(2,2,figsize=(175/25.4,145/25.4))
        fig.subplots_adjust(left=.12,right=.96,bottom=.12,top=.91,hspace=.84,wspace=.59)
        A=np.array([38.4842,38.1546,38.0838,37.7695,37.9597,37.7434,37.9936,37.9034,38.2161,38.0199])
        B=np.array([37.6447,37.7790,37.5484,37.6745,37.6028,38.2678,38.2853,37.6207,37.8422,38.0286])
        test=f.read('baseline_bottle_test').iloc[0]
        ax=axs[0,0];f.panel(ax,'A','Initial bottle weight')
        for i,values in enumerate([A,B]):
            ax.bar(i,values.mean(),.6,color='#6688b9',yerr=f.stats.sem(values),capsize=3,zorder=2)
            ax.scatter(i+f.rng_for('bottle'+str(i)).normal(0,.055,len(values)),values,s=11,c='.25',alpha=.35,zorder=3)
        f.bracket(ax,0,1,38.61,.10,'ns' if test.p>=.05 else f.stars(test.p),fontsize=8)
        ax.set(xticks=[0,1],xticklabels=['Bottle 1','Bottle 2'],ylabel='Weight (g)',ylim=(36,39.1),yticks=[36,37,38,39],xlim=(-.6,1.6))
        ax=axs[0,1];f.panel(ax,'B','Total intake')
        pairs=f.read('age_matched_intake_pairs')
        for sex,offset,color in [('M',-.18,f.GRAY),('F',.18,f.PINK)]:
            g=pairs[pairs.Sex.eq(sex)]
            for j,column in enumerate(['water_total_g','palatable_total_g']):
                values=g[column].to_numpy();x=j+offset
                ax.bar(x,values.mean(),.32,color=color,yerr=f.stats.sem(values),capsize=3,label=sex if j==0 else None,zorder=2)
                ax.scatter(x+f.rng_for(sex+column).normal(0,.035,len(values)),values,s=10,c=color,edgecolors='.3',linewidths=.2,alpha=.65,zorder=3)
        ax.set(xticks=[0,1],xticklabels=['Water only','Palatable'],ylabel='Session intake (g)',ylim=(0,max(pairs[['water_total_g','palatable_total_g']].max())*1.12))
        ax.legend(frameon=False,ncol=2,loc='upper left',handlelength=.8,columnspacing=.8)
        gr=f.read('group_cosinor').set_index('group');x=np.arange(4)
        ax=axs[1,0];f.panel(ax,'C','Phase organization')
        for sex,offset,color in [('M',-.18,f.GRAY),('F',.18,f.PINK)]:
            v=gr.reindex([f'{sex} {age}' for age in f.AGE_ORDER])
            ax.bar(x+offset,v.p_phase_permutation,.32,color=color,label=sex)
            ax.scatter(x+offset,v.dominance_overlap,s=19,marker='D',facecolors='white',edgecolors=color,linewidths=1,zorder=4)
        ax.axhline(.05,c='.5',ls='--',lw=1)
        ax.set(xticks=x,xticklabels=['Young','Early-mid','Middle','Old'],ylabel='Permutation p',ylim=(0,.65),yticks=[0,.2,.4,.6])
        ax.tick_params(axis='x',rotation=25);ax.legend(frameon=False,ncol=2,loc='upper left',handlelength=.8,columnspacing=.8)
        ind=f.read('individual_cosinor');ax=axs[1,1];f.panel(ax,'D','Individual rhythms')
        for sex,offset,color in [('M',-.18,f.GRAY),('F',.18,f.PINK)]:
            nominal=[];adjusted=[]
            for age in f.AGE_ORDER:
                g=ind[ind.legacy_age_cohort & ind.metric.eq('preference') & ind.Sex.eq(sex) & ind.age_group.eq(age)]
                nominal.append(100*np.mean(g.p<.05));adjusted.append(100*np.mean(g.q<.05))
            ax.bar(x+offset,nominal,.32,color=color)
            ax.scatter(x+offset,adjusted,marker='_',s=80,c='k',linewidths=1.3,zorder=4)
        ax.axhline(5,c='.5',ls='--',lw=1)
        ax.set(xticks=x,xticklabels=['Young','Early-mid','Middle','Old'],ylabel='Rhythmic sessions (%)',ylim=(-1,15.5),yticks=[0,5,10,15])
        ax.tick_params(axis='x',rotation=25)
        f.save(fig,'FigureS2')

    def supplement3():
        bins=f.pd.read_csv(f.OUT/'data/analysis_bins.csv')
        selection=f.read('landscape_selection');mid=selection.iloc[0].MouseID
        g=bins[bins.MouseID.eq(mid)].sort_values('bin_index')
        p=f.read('corrected_expected_model_predictions');p=p[p.MouseID.eq(mid)&p.model.eq('simple')].sort_values('bin_index')
        assert len(g)==len(p)==8
        t=g.elapsed_h.to_numpy();n=p.N_bin.to_numpy();prob=p.predicted.to_numpy()
        fig=f.plt.figure(figsize=(175/25.4,145/25.4))
        positions=[[.10,.735,.20,.18],[.405,.735,.20,.18],[.71,.735,.20,.18]]
        ax=fig.add_axes(positions[0]);f.panel(ax,'A','Assumed events')
        ax.plot(t,n,'o-',c='k',ms=3,lw=1.8);ax.fill_between(t,0,n,color='.92');ax.set(ylabel='Events per interval',ylim=(0,max(n)*1.18))
        ax=fig.add_axes(positions[1]);ax.set_title('Expected choices')
        ax.plot(t,n*prob,'o-',c=f.ORANGE,ms=3,lw=1.8,label='Sucrose')
        ax.plot(t,n*(1-prob),'o-',c=f.BLUE,ms=3,lw=1.8,label='Water');ax.set(ylabel='Choices per interval',ylim=(0,max(n)*1.18))
        ax=fig.add_axes(positions[2]);ax.set_title('Preference')
        ax.plot(t,prob,'o-',c='#e62b2b',ms=3,lw=1.8);ax.axhline(.5,c='.55',ls='--',lw=.8);ax.set(ylabel='Sucrose / total',ylim=(0,1),yticks=[0,.5,1])
        for ax in fig.axes:
            ax.set(xlim=(0,24),xticks=[0,12,24],xlabel='Elapsed time (h)');ax.axvspan(12,24,color='.93',zorder=-5)
        fig.legend(handles=[Line2D([0],[0],color=f.ORANGE,marker='o',label='Sucrose'),Line2D([0],[0],color=f.BLUE,marker='o',label='Water')],loc='center',bbox_to_anchor=(.51,.664),ncol=2,frameon=False,handlelength=1.1)
        ax=fig.add_axes([.18,.485,.65,.095]);f.panel(ax,'B','Bottle positions')
        for side in [0,1]:
            for i in range(8):
                sweet=(i+side)%2==0
                ax.add_patch(Rectangle((i*3,side),3,1,facecolor=f.ORANGE if sweet else f.BLUE,edgecolor='white',lw=.6))
        ax.set(xlim=(0,24),ylim=(0,2),yticks=[.5,1.5],yticklabels=['Left','Right'],xticks=[0,6,12,18,24],xlabel='Elapsed time (h)')
        ax=fig.add_axes([.10,.055,.81,.315]);f.panel(ax,'C','Model structure');ax.axis('off')
        rows=[['Random','0','Fixed preference of 0.5'],['Bias','1','Constant fitted preference'],['Linear / day–night','2','Drift or day–night step'],['Simple RL','3','Learning; constant reward and sensitivity'],['Reward RL','5','Learning; rhythmic reward'],['Full RL','6','Learning; rhythmic reward and sensitivity']]
        table=ax.table(cellText=rows,colLabels=['Model','Parameters','Mechanism'],colWidths=[.24,.17,.59],bbox=[0,0,1,1],cellLoc='left')
        table.auto_set_font_size(False);table.set_fontsize(8)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('#d9d9d9');cell.set_linewidth(.6);cell.PAD=.06
            if row==0:cell.set_facecolor('#eeeeee');cell.set_text_props(weight='bold')
        f.save(fig,'FigureS3')

    f.supplement1=supplement1;f.supplement2=supplement2;f.supplement3=supplement3
