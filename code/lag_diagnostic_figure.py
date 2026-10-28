"""Transparent supporting display for the revised Figure2F analysis."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build(base,dest):
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.linewidth':1.1,'xtick.major.width':1.1,'ytick.major.width':1.1,'pdf.fonttype':42,'svg.fonttype':'none'})
    units=pd.read_csv(base/'figure2F_paired_lag_animals.csv')
    summary=pd.read_csv(base/'figure2F_paired_lag_summary.csv').set_index('scope')
    xc=pd.read_csv(base/'intake_preference_xcorr_summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(175/25.4,85/25.4),gridspec_kw={'width_ratios':[1.05,1]})
    fig.subplots_adjust(left=.17,right=.98,top=.86,bottom=.25,wspace=.48)
    ax=axs[0];rg=np.random.default_rng(20261027)
    for y,(scope,label,color) in enumerate([('Control_age_all','All','#3478a3'),('Control_age_F','Female','#b8679f'),('Control_age_M','Male','#555555')]):
        g=units if label=='All' else units[units.Sex.eq('F' if label=='Female' else 'M')]
        s=summary.loc[scope]
        ax.scatter(g.lag_h,y+rg.uniform(-.15,.15,len(g)),s=6,alpha=.25,c=color,lw=0)
        ax.errorbar(s.mean_lag_h,y,xerr=[[s.mean_lag_h-s.ci_low_h],[s.ci_high_h-s.mean_lag_h]],fmt='o',c='k',mfc='white',ms=4,lw=1.5,capsize=3,zorder=4)
    ax.axvline(0,c='.55',ls='--',lw=.8);ax.set(yticks=[0,1,2],yticklabels=['All (113)','Female (52)','Male (61)'],ylim=(2.6,-.6),xlim=(-12,12),xticks=[-12,-6,0,6,12],xlabel='Preference − intake acrophase (h)',title='A  Paired phase differences')
    ax=axs[1]
    for scope,color,label in [('All','#3478a3','All'),('F','#b8679f','Female'),('M','#555555','Male')]:
        g=xc[xc.scope.eq(scope)].sort_values('lag_h');x=np.r_[g.lag_h,12];m=np.r_[g.mean_r,g.mean_r.iloc[0]]
        if scope=='All':ax.fill_between(x,np.r_[g.simultaneous_low,g.simultaneous_low.iloc[0]],np.r_[g.simultaneous_high,g.simultaneous_high.iloc[0]],color=color,alpha=.14,lw=0)
        ax.plot(x,m,'o-',c=color,ms=2.5,lw=1.2 if scope=='All' else .8,label=label,alpha=1 if scope=='All' else .7)
    ax.axhline(0,c='.7',lw=.7);ax.axvline(0,c='.55',ls='--',lw=.8);ax.set(xlim=(-12,12),xticks=[-12,-6,0,6,12],ylim=(-.3,.3),yticks=[-.2,0,.2],xlabel='Preference lag relative to intake (h)',ylabel='Mean within-session Pearson r',title='B  Periodic profile cross-correlation')
    ax.legend(frameon=False,loc='upper left',fontsize=8,handlelength=1.2,labelspacing=.2)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.text(.55,.055,'Positive lag: preference follows intake. Bars: 95% animal-bootstrap intervals.\nBlue band: simultaneous 95% interval across eight unique 3-h lags.',ha='center',fontsize=8)
    dest.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:fig.savefig(dest/('Figure2_phase_lag_checks.'+ext),dpi=300)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();build(a.base,a.dest)
