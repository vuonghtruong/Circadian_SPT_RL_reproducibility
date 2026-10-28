"""One entry point assembles all manuscript figures from generated tables."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle,Ellipse
from matplotlib.transforms import ScaledTranslation
import json,tempfile
from publication_style import prepare,WIDTH_IN,DPI
from statistics import cosinor,permutation_difference
from models import predict,allocate,bounds_for,PARAM_NAMES

BLUE='#2878ad';ORANGE='#d67a23';PINK='#c94f87';GRAY='#65676b';GREEN='#46968c'
AGE_COLOR={'Young':'#3976a9','EarlyMid':'#59a89c','Mid':'#d99135','Old':'#a45b98'}
MODEL_COLOR={'random':'#a0a0a0','bias':'#585858','linear_bias':'#aa75ad','step_bias':'#75a89e','simple':BLUE,'simple_no_swap':'#77afca','reward':ORANGE,'reward_no_swap':'#e9b48a','full':'#b14750','full_no_swap':'#d9979e'}
LABEL={'random':'Random','bias':'Bias','linear_bias':'Linear bias','step_bias':'Day/night bias','simple':'Simple','simple_no_swap':'Simple NS','reward':'Reward','reward_no_swap':'Reward NS','full':'Full','full_no_swap':'Full NS'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.titlelocation':'center','axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def read(name):return pd.read_csv(OUT/'results'/f'{name}.csv')
def stars(p):return 'n.s.' if not np.isfinite(p) or p>=.05 else '***' if p<.001 else '**' if p<.01 else '*'
def bracket(ax,x1,x2,y,h,label,color='k',fontsize=7):
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y],c=color,lw=.8,clip_on=False)
    ax.text((x1+x2)/2,y+h,label,ha='center',va='bottom',fontsize=fontsize,color=color)
def individual_lines(ax,g,metric,color,alpha=.15):
    for _,d in g.groupby('MouseID'):
        d=d.sort_values('ZT');ax.plot(d.ZT,d[metric],c=color,lw=.45,alpha=alpha,zorder=1)
def panel(ax,letter,title=''):
    if letter:
        transform=ax.transAxes+ScaledTranslation(-18/72,10/72,ax.figure.dpi_scale_trans)
        t=ax.text(0,1,letter,transform=transform,fontweight='bold',fontsize=11,va='bottom')
        t._panel_letter=True
    if title:ax.set_title(title,loc='center',pad=8)
def save(fig,name):
    prepare(fig,name)
    for ext in ['png','svg','pdf']:
        target=OUT/'figures'/f'{name}.{ext}'
        with tempfile.NamedTemporaryFile(dir=target.parent,suffix='.'+ext,delete=False) as handle:
            temporary=Path(handle.name)
        try:
            fig.savefig(temporary,format=ext,dpi=DPI,bbox_inches=None,metadata={'Creator':'Circadian SPT manuscript revision'} if ext=='pdf' else None)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)
def shade(ax):ax.axvspan(12,24,color='k',alpha=.055,lw=0);ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24])
def profile(ax,g,metric='preference',color=ORANGE,fit=False,variation=False,label=None):
    s=g.groupby('ZT')[metric].agg(['mean','sem',lambda v:v.quantile(.25),lambda v:v.quantile(.75)]).sort_index()
    s.columns=['mean','sem','q25','q75'];t=s.index.to_numpy()
    if variation:ax.fill_between(t,s.q25,s.q75,color=color,alpha=.12,lw=0)
    ax.errorbar(t,s['mean'],yerr=s['sem'],color=color,fmt='o-',ms=2.5,lw=1,capsize=1.5,label=label)
    if fit:
        c=cosinor(t,s['mean']);xx=np.linspace(0,24,200);yy=c['mesor']+c['beta_cos']*np.cos(xx*np.pi/12)+c['beta_sin']*np.sin(xx*np.pi/12);ax.plot(xx,yy,color=color,lw=1.4,ls='--')
    shade(ax);ax.set_xlabel('Source ZT/CT (h)')
    if metric=='preference':ax.set_ylim(0,1);ax.set_ylabel('Preference')
    else:ax.set_ylabel('Intake (g / 3 h)')
def polar(ax,phases,identified=None):
    p=np.asarray(phases);good=np.isfinite(p);p=p[good];theta=p*np.pi/12
    ax.set_theta_zero_location('N');ax.set_theta_direction(-1);ax.set_xticks(np.arange(4)*np.pi/2,['0','6','12','18']);ax.set_ylim(0,1.22);ax.set_yticks([.5,1]);ax.set_yticklabels([]);ax.grid(alpha=.25)
    if len(p):
        significant=np.zeros(len(p),bool) if identified is None else np.asarray(identified)[good]
        ax.scatter(theta[~significant],np.ones(sum(~significant)),s=12,facecolors='none',edgecolors='#aaaaaa',lw=.6,clip_on=False)
        ax.scatter(theta[significant],np.ones(sum(significant)),s=15,c=ORANGE,clip_on=False)
        mu=mean_phase(p)*np.pi/12;R=resultant(p);ax.annotate('',xy=(mu,R),xytext=(mu,0),arrowprops={'arrowstyle':'->','color':ORANGE,'lw':1.6})

def figure1(bins):
    ind=read('individual_cosinor');gr=read('group_cosinor').set_index('group');age=bins[bins.legacy_age_cohort]
    fig=plt.figure(figsize=(WIDTH_IN,9.1));gs=fig.add_gridspec(4,5,width_ratios=[2,1.5,1,1.5,1],left=.09,right=.94,bottom=.085,top=.94,hspace=.8,wspace=.40)
    ax=fig.add_subplot(gs[0,0]);ax.axis('off');panel(ax,'A','Two-bottle assay')
    for i,text_ in enumerate(['5-day habituation','24-h preference session','Weigh and swap every 3 h']):
        ax.text(.5,.85-i*.33,text_.replace(' ','\n',1) if i<2 else 'Weigh and swap\nevery 3 h',ha='center',va='center',transform=ax.transAxes,bbox={'boxstyle':'round,pad=.3','facecolor':'#f0f3f5','edgecolor':'#cbd1d5'})
        if i<2:ax.annotate('',(.5,.66-i*.33),(.5,.73-i*.33),xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
    ax=fig.add_subplot(gs[1,0]);panel(ax,'B','Fluid preference')
    data=read('figure1B_animal_means');tests=read('figure1B_vs_water_tests').set_index('fluid');rg=rng_for('tastantpoints')
    for i,fluid in enumerate(['water','glucose','sucrose','saccharin']):
        v=data[data.fluid==fluid].preference.dropna();ax.bar(i,v.mean(),width=.65,color='#6688b9',alpha=.9,yerr=stats.sem(v),capsize=3,error_kw={'lw':.9});ax.scatter(rg.normal(i,.065,len(v)),v,s=8,c='#252525',alpha=.3);ax.text(i,1.04,str(len(v)),ha='center',fontsize=8)
        if fluid!='water':ax.text(i,.91,stars(tests.loc[fluid,'q_bh']),ha='center',fontsize=8)
    ax.axhline(.5,ls='--',c='gray',lw=.8);ax.set_xticks(range(4),['Water A','Glucose','Sucrose','Saccharin'],rotation=25,ha='right');ax.set_ylabel('Preference');ax.set_ylim(0,1.12)
    ax.text(.5,.02,'BH FDR vs water',transform=ax.transAxes,ha='center',fontsize=8)
    for row,(letter,column,title) in enumerate([('C','p_group_mean_OLS','Group cosinor p'),('D','emergence','Phase emergence')],start=2):
        ax=fig.add_subplot(gs[row,0]);panel(ax,letter,title)
        for sex,color,offset in [('F',PINK,-.17),('M',GRAY,.17)]:
            v=[gr.loc[f'{sex} {a}',column] for a in AGE_ORDER];ax.bar(np.arange(4)+offset,v,.32,color=color,label=sex)
        ax.set_xticks(range(4),['Young','Early-mid','Mid','Old'],rotation=25,ha='right');ax.set_ylabel('Cosinor p (unadjusted)' if row==2 else 'Observed R / mean null R');ax.axhline(.05 if row==2 else 1,c='k',ls='--',lw=.7)
        if row==2:ax.set_yscale('log');ax.set_ylim(.001,1);ax.legend(frameon=False)
    for sex,c,letters in [('F',1,'EFGH'),('M',3,'IJKL')]:
        for j,a in enumerate(AGE_ORDER):
            g=age[(age.Sex==sex)&(age.age_group==a)];ax=fig.add_subplot(gs[j,c]);panel(ax,letters[j],f'{sex}, {a.replace("EarlyMid","Early-mid")} (n={g.MouseID.nunique()})')
            for _,individual in g.groupby('MouseID'):
                individual=individual.sort_values('ZT');ax.plot(individual.ZT,individual.preference,c='#8b8b8b',alpha=.28,lw=.55,zorder=1)
            profile(ax,g,color=BLUE);ax.set_xticks([0,12,24]);ax.set_xlabel('ZT/CT (h)')
            ax.set_ylabel('')
            s=g.groupby('ZT').preference.mean();cf=cosinor(s.index,s.values);xx=np.linspace(0,24,250)
            ax.plot(xx,cf['mesor']+cf['beta_cos']*np.cos(xx*np.pi/12)+cf['beta_sin']*np.sin(xx*np.pi/12),c=ORANGE,lw=1.8,zorder=5)
            ax=fig.add_subplot(gs[j,c+1],projection='polar');ss=ind[(ind.metric=='preference')&(ind.MouseID.isin(g.MouseID))];polar(ax,[])
            res=read('acrophase_resamples_'+sex+'_'+a)
            for method,color,limit,alpha in [('permutation','#777777',1000,.09),('bootstrap',BLUE,500,.14)]:
                q=res[res.method==method].head(limit);ax.scatter(q.phase_h*np.pi/12,q.R,s=2,c=color,alpha=alpha,lw=0)
            valid=ss.phase_h.notna();sig=ss.p<.05
            ax.scatter(ss.loc[valid&~sig,'phase_h']*np.pi/12,np.ones(sum(valid&~sig)),s=11,facecolors='none',edgecolors=ORANGE,alpha=.6,lw=.6,clip_on=False)
            ax.scatter(ss.loc[valid&sig,'phase_h']*np.pi/12,np.ones(sum(valid&sig)),s=13,c=ORANGE,clip_on=False)
            mu=mean_phase(ss.phase_h)*np.pi/12;R=resultant(ss.loc[valid,'phase_h']);ax.annotate('',xy=(mu,R),xytext=(mu,0),arrowprops={'arrowstyle':'->','color':ORANGE,'lw':1.6})
    fig.text(.5,.008,'Profiles: preference, individual sessions (gray), mean ± SEM (blue), group cosinor (orange). Circular clouds: 500 bootstrap (blue), 1,000 permutation (gray). B: counts above bars indicate n.',ha='center',fontsize=8)
    save(fig,'Figure1')

def figure2(bins):
    age=bins[bins.legacy_age_cohort];phase=read('paired_phases');dn=read('day_night_tests');lag=read('paired_circular_tests')
    fig=plt.figure(figsize=(WIDTH_IN,10.1))
    gs=fig.add_gridspec(5,4,height_ratios=[1.65,1.65,1.25,2.4,1.65],left=.10,right=.91,top=.955,bottom=.09,hspace=.95,wspace=.72)
    for row,sex in enumerate(['F','M']):
        top=gs[row,:].subgridspec(1,4,wspace=.32)
        for j,a in enumerate(AGE_ORDER):
            g=age[(age.Sex==sex)&(age.age_group==a)];cell=top[j].subgridspec(2,1,height_ratios=[1,.8],hspace=.12);ax=fig.add_subplot(cell[0]);panel(ax,'A' if row==0 and j==0 else 'B' if row==1 and j==0 else '',f'{a.replace("EarlyMid","Early-mid")}\n{sex}, n={g.MouseID.nunique()}')
            individual_lines(ax,g,'total_g',BLUE,.22);profile(ax,g,'total_g',BLUE);ax.set_ylim(0,5);twin=ax.twinx();individual_lines(twin,g,'preference',ORANGE,.22);profile(twin,g,'preference',ORANGE);ax.tick_params(axis='y',colors=BLUE);twin.tick_params(axis='y',colors=ORANGE)
            ax.set_xlabel('');ax.tick_params(labelbottom=False);twin.set_xlabel('')
            ax.set_ylabel('Total\nintake (g)' if j==0 else '',color=BLUE);twin.set_ylabel('Preference' if j==3 else '',color=ORANGE)
            if j:ax.tick_params(labelleft=False)
            if j!=3:twin.tick_params(labelright=False)
            lower=fig.add_subplot(cell[1]);
            for metric,color,label in [('water_g','#3986b6','Water'),('reward_g','#c76c50','Sweet')]:
                individual_lines(lower,g,metric,color,.22);profile(lower,g,metric,color,label=label)
            lower.set_ylim(0,4);lower.set_yticks([0,2,4]);lower.set_xticks([0,12,24]);lower.set_ylabel('Fluid\nintake (g)' if j==0 else '');lower.set_xlabel('ZT/CT (h)')
            if j:lower.tick_params(labelleft=False)
            if j==0:lower.legend(frameon=False,fontsize=8,loc='upper left',handlelength=1,handletextpad=.3,labelspacing=.15)
    daytests=read('figure2C_tests').set_index('metric');ax=fig.add_subplot(gs[2,0]);panel(ax,'C',f'Day/night (n={int(daytests.loc["total_g","n"])})')
    for i,(metric,color) in enumerate([('total_g',BLUE),('preference',ORANGE)]):
        row=daytests.loc[metric];target=ax if i==0 else ax.twinx()
        if i==0:target.bar([0,1],[row.day_mean,row.night_mean],yerr=[row.day_sem,row.night_sem],width=.62,color=['#bbb','#555'],capsize=3)
        else:target.errorbar([0,1],[row.day_mean,row.night_mean],yerr=[row.day_sem,row.night_sem],fmt='o-',color=color,capsize=3,ms=4)
        target.set_ylim(0,4 if i==0 else 1.2);target.tick_params(axis='y',colors=color)
        target.set_ylabel('Intake (g / 3 h)' if i==0 else '',color=color)
        bracket(target,0,1,3.35 if i==0 else .84,.05 if i==0 else .02,stars(row.q_bh),color='k' if i==0 else color,fontsize=8)
    ax.set_xticks([0,1],['Day','Night']);ax.set_xlim(-.25,1.25)
    for column,metric,letter in [(1,'total_phase','D'),(2,'pref_phase','E')]:
        ax=fig.add_subplot(gs[2,column]);panel(ax,letter,'Intake phase (h)' if metric=='total_phase' else 'Preference phase (h)')
        for sex,color in [('F',PINK),('M',GRAY)]:
            g=phase[phase.legacy_age_cohort & phase.Sex.eq(sex)];ax.scatter(g.Age,g[metric],s=14,c=color,alpha=.55,label=sex)
            trend=read('figure2DE_trend_lines');trend=trend[trend.metric.eq(metric)&trend.Sex.eq(sex)];ax.plot(trend.Age,trend['mean'],color=color,lw=1.3);ax.fill_between(trend.Age,trend.low,trend.high,color=color,alpha=.12,lw=0)
        ax.set_ylim(0,24);ax.set_yticks([0,12,24]);ax.set_xlabel('Age (days)');ax.set_ylabel('')
        if letter=='D':sex_handles,sex_labels=ax.get_legend_handles_labels()
    fig.legend(sex_handles,['Female','Male'],frameon=False,ncol=2,loc='center',bbox_to_anchor=(.515,gs[2,1:3].get_position(fig).y1+.044),handlelength=.7,columnspacing=1.)
    ax=fig.add_subplot(gs[2,3]);panel(ax,'F','Bootstrap phase\nconcentration')
    curve=read('intake_preference_xcorr_summary');curve=curve[curve.scope.eq('All')].sort_values('lag_h')
    # +/-12h are the same periodic displacement, displayed at both boundaries.
    x=np.r_[curve.lag_h.to_numpy(),12];mean=np.r_[curve.mean_r.to_numpy(),curve.mean_r.iloc[0]]
    lo=np.r_[curve.simultaneous_low.to_numpy(),curve.simultaneous_low.iloc[0]];hi=np.r_[curve.simultaneous_high.to_numpy(),curve.simultaneous_high.iloc[0]]
    ax.fill_between(x,lo,hi,color=BLUE,alpha=.16,lw=0)
    ax.axhline(0,c='.65',lw=.7);ax.axvline(0,c='.65',ls='--',lw=.7)
    ax.plot(x,mean,'o-',c=BLUE,ms=2.1,lw=1.1)
    ax.set(xlim=(-12,12),ylim=(-.25,.25),xlabel='Lag (h)',ylabel='Mean r',xticks=[-12,0,12],yticks=[-.2,0,.2])
    ax.text(.98,.96,'n = 113',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    summary=[]
    for i,sex in enumerate(['F','M']):
        ax=fig.add_subplot(gs[3,2*i:2*i+2]);panel(ax,'G' if i==0 else 'H','Female' if sex=='F' else 'Male')
        for x,y,color in [(0,0,'#ffffff'),(12,0,'#eeeeee'),(0,12,'#eeeeee'),(12,12,'#d3d3d3')]:ax.add_patch(Rectangle((x,y),12,12,facecolor=color,lw=0,zorder=0))
        ax.axvline(12,c='#aaa',ls=':',lw=.6);ax.axhline(12,c='#aaa',ls=':',lw=.6);ax.plot([0,24],[0,24],'--',c='gray',lw=.9)
        for a in AGE_ORDER:
            g=phase[phase.legacy_age_cohort & phase.Sex.eq(sex)&phase.age_group.eq(a)].dropna(subset=['total_phase','pref_phase']);x=g.total_phase.to_numpy();y=g.pref_phase.to_numpy();color=AGE_COLOR[a]
            ax.scatter(x,y,s=14,c=color,alpha=.7,label=a.replace('EarlyMid','Early-mid'))
            mx,my=mean_phase(x),mean_phase(y);sx=stats.sem(wrap_hours(x-mx));sy=stats.sem(wrap_hours(y-my))
            ax.scatter(mx,my,s=48,facecolors='white',edgecolors=color,lw=1.2,zorder=5)
            for dx in [-24,0,24]:
                for dy in [-24,0,24]:ax.add_patch(Ellipse((mx+dx,my+dy),4*sx,4*sy,fill=False,edgecolor=color,ls='--',lw=.9,alpha=.85))
            summary.append(dict(Sex=sex,age_group=a,n=len(g),intake_circular_mean_h=mx,preference_circular_mean_h=my,intake_wrapped_SEM_h=sx,preference_wrapped_SEM_h=sy))
        ax.set(xlim=(0,24),ylim=(0,24),xlabel='Total intake\nacrophase (h)',ylabel='Preference\nacrophase (h)' if i==0 else '');ax.set_aspect('equal',adjustable='box');ax.set_xticks([0,12,24]);ax.set_yticks([0,12,24])
        if i==0:
            handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,frameon=False,fontsize=8,loc='center',bbox_to_anchor=(.515,gs[3,:].get_position(fig).y1+.035),handlelength=.6,handletextpad=.4,ncol=4,columnspacing=.8)
    save_table(pd.DataFrame(summary),'figure2_phase_means_sem')
    colors={'Control':'#ed9418','PA':'#ef5558','RF-morning':'#63c849','RF-afternoon':'#269b45'};markers={'Control':'o','PA':'*','RF-morning':'s','RF-afternoon':'^'}
    bottom=gs[4,:].subgridspec(1,3,width_ratios=[1.15,1.1,1.1],wspace=.38);legend=fig.add_subplot(bottom[0]);legend.axis('off');legend.set(xlim=(0,1),ylim=(0,1));legend.text(.5,1.05,'Entrainment condition',fontsize=9,ha='center')
    ph=read('figure2IJ_phase_points');comparisons=read('figure2IJ_tests')
    for j,(condition,color) in enumerate(colors.items()):
        yy=.91-j*.20;legend.scatter(.025,yy,c=color,marker=markers[condition],s=45,clip_on=False);legend.text(.12,yy,condition+'  n='+str(sum(ph.Condition_real.eq(condition))),va='center',fontsize=8)
    legend.text(0,.09,'First session / condition.\nAll pairs versus Control.\nPA reference unverified.',fontsize=8,va='top')
    for i,(col,letter,title) in enumerate([('total_phase','I','Total intake'),('pref_phase','J','Preference')]):
        ax=fig.add_subplot(bottom[i+1],projection='polar');panel(ax,letter,title);polar(ax,[]);ax.set_ylim(0,1.27);ax.set_yticks([]);ax.set_yticklabels([]);ax.spines['polar'].set_visible(False);ax.grid(False)
        for angle in np.arange(4)*np.pi/2:ax.plot([angle,angle],[0,1],color='#dddddd',lw=.6,zorder=0)
        theta=np.linspace(np.pi,2*np.pi,100);ax.fill_between(theta,0,1,color='#ededed',zorder=0);ax.plot(np.linspace(0,2*np.pi,200),np.ones(200),c='k',lw=.8)
        means={}
        for j,(condition,color) in enumerate(colors.items()):
            p=ph[ph.Condition_real.eq(condition)][col].dropna();mu=mean_phase(p)*np.pi/12;R=resultant(p);means[condition]=mu
            radius=rng_for(col+condition+'polar').uniform(1.03,1.18,len(p));ax.scatter(p*np.pi/12,radius,c=color,marker=markers[condition],s=11,alpha=.6,lw=0,clip_on=False)
            ax.plot([mu,mu],[0,R],c=color,lw=1.3);ax.scatter(mu,R,c=color,marker=markers[condition],s=60,zorder=5,edgecolors='white',lw=.5)
        pair_labels=[]
        for j,condition in enumerate(['PA','RF-morning','RF-afternoon']):
            start=means['Control'];diff=(means[condition]-start+np.pi)%(2*np.pi)-np.pi;tt=np.linspace(start,start+diff,80);radius=.21+.21*j;test=comparisons[comparisons.metric.eq(col)&comparisons.condition.eq(condition)].iloc[0]
            color=colors[condition]
            ax.plot(tt,np.full(len(tt),radius),c=color,lw=.95);ax.plot([tt[0],tt[0]],[radius-.035,radius+.035],c=color,lw=.95);ax.plot([tt[-1],tt[-1]],[radius-.035,radius+.035],c=color,lw=.95)
            short={'PA':'PA','RF-morning':'AM','RF-afternoon':'PM'}[condition]
            pair_labels.append(f'{short}: {stars(test.q_bh).replace("n.s.","ns")}')
        # Keep all three comparison results clear of mean vectors and the
        # individual phase points; each arc uses its condition's legend color.
        ax.text(.5,-.29,'; '.join(pair_labels),transform=ax.transAxes,ha='center',va='top',fontsize=8,clip_on=False)
    fig.text(.5,.008,'A/B: individuals and mean ± SEM. F: periodic cross-correlation, positive lag preference follows intake;95% simultaneous animal-bootstrap band.\nD/E and phase comparisons are descriptive.',ha='center',fontsize=8)
    save(fig,'Figure2')

def figure3():
    simulation=read('illustrative_profiles');dn=read('illustrative_day_night');fits=read('corrected_expected_model_fits');traces=read('illustrative_event_trajectories')
    models=['random','bias','simple','simple_no_swap','reward','reward_no_swap','full','full_no_swap']
    A=fits.pivot(index='MouseID',columns='model',values='pseudo_AIC_1000')[models];best=A.idxmin(axis=1)
    priority=['bias','simple','simple_no_swap','reward','reward_no_swap','full','full_no_swap','random']
    order=best.map({m:i for i,m in enumerate(priority)}).sort_values(kind='stable').index;delta=A.sub(A.min(axis=1),axis=0)
    fig=plt.figure(figsize=(WIDTH_IN,11.3));gs=fig.add_gridspec(6,12,height_ratios=[1,1,1,1.2,1.25,1.25],left=.14,right=.89,top=.94,bottom=.105,hspace=.95,wspace=1.2)
    ax=fig.add_subplot(gs[0,:3]);panel(ax,'A','Day (light),\nnight (dark)')
    five=['random','bias','simple','reward','full']
    for i,phase_ in enumerate(['Day','Night']):
        sm=dn[dn.phase==phase_].groupby('model').preference.agg(['mean','sem']).reindex(five)
        ax.bar(np.arange(5)+(i-.5)*.32,sm['mean'],.32,yerr=sm['sem'],capsize=2,color='#aaa' if i==0 else '#333',label=phase_)
    tests=read('figure3A_tests').set_index('model')
    for i,m in enumerate(five):bracket(ax,i-.16,i+.16,.86,.035,stars(tests.loc[m,'q_bh']),fontsize=6)
    ax.set_xticks(range(5),['Random','Bias','Simple','Reward','Full'],rotation=60,ha='right');ax.set_ylim(0,1.04);ax.set_ylabel('Preference')
    ax=fig.add_subplot(gs[0,3:6]);panel(ax,'B','Mean ± SEM\n500 runs')
    for m in five:
        d=simulation[simulation.model==m];ax.errorbar(d.ZT,d['mean'],yerr=d['sem'],fmt='o-',color=MODEL_COLOR[m],label=LABEL[m],ms=2,lw=.9,capsize=2,elinewidth=.7)
    shade(ax);ax.set_ylim(0,1);ax.set_xticks([0,12,24]);ax.set_xlabel('ZT (h)');ax.set_ylabel('')
    placements=[('simple','E',gs[0,6:]),('random','C',gs[1,:6]),('reward','F',gs[1,6:]),('bias','D',gs[2,:6]),('full','G',gs[2,6:])]
    for model,letter,position in placements:
        ax=fig.add_subplot(position);panel(ax,letter,LABEL[model]);d=traces[traces.model==model]
        ax.plot(d.time_h,d.rolling_choice,c=MODEL_COLOR[model],lw=1.0)
        shade(ax);ax.set_ylim(-.03,1.03);ax.set_yticks([0,1],['Water','Sweet']);ax.set_xticks(np.arange(0,25,3));ax.set_xlabel('ZT (h)')
        if letter in ['E','F','G']:ax.yaxis.tick_right()
    ax=fig.add_subplot(gs[3,:6]);panel(ax,'H','Mean score difference')
    means=delta.mean().sort_values();se=delta.sem().reindex(means.index);ax.bar(range(len(means)),means,yerr=se,color=plt.cm.Greys(np.linspace(.85,.3,len(means))),capsize=2);ax.set_xticks(range(len(means)),[LABEL[m] for m in means.index],rotation=30,ha='right');ax.set_ylabel('Mean delta pseudo-AIC')
    ax=fig.add_subplot(gs[3,6:]);panel(ax,'I',f'Winning count, best to worst (n={len(A)})');c=best.value_counts().reindex(priority,fill_value=0).sort_values(ascending=False,kind='stable')
    ax.bar(range(8),c,color=plt.cm.Greys(np.linspace(.85,.3,8)));ax.set_xticks(range(8),[LABEL[m] for m in c.index],rotation=30,ha='right');ax.set_ylabel('Likely animals');ax.set_ylim(0,max(c)*1.22)
    for i,v in enumerate(c):ax.text(i,v+max(c)*.025,str(v),ha='center',fontsize=7)
    boundaries=np.flatnonzero(best.loc[order].to_numpy()[1:]!=best.loc[order].to_numpy()[:-1])+.5
    for row,letter,title,matrix,cmap,vmin,vmax,barlabel in [(4,'J','Score differences across animals',delta,'viridis_r',0,100,'Delta score; capped at 100')]:
        ax=fig.add_subplot(gs[row,:]);panel(ax,letter,title);im=ax.imshow(matrix.loc[order].T,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest');ax.set_yticks(range(8),[LABEL[m] for m in models]);ticks=np.unique(np.linspace(0,len(order)-1,7,dtype=int));ax.set_xticks(ticks,ticks+1);ax.set_xlabel('Animal ordered by best-fitting class')
        for edge in boundaries:ax.axvline(edge,c='white',lw=.7,alpha=.75)
        fig.colorbar(im,ax=ax,fraction=.016,pad=.015,label=barlabel)
    fig.text(.5,.012,'C-G: one simulated event sequence, 31-event moving mean; not measured licks. N=1,000 assumed events (200 day / 800 night). NS: no swap.',ha='center',fontsize=6.5)
    save_table(pd.DataFrame({'plot_order':np.arange(1,len(order)+1),'MouseID':order,'best_model':best.loc[order].values}),'figure3_animal_order')
    save(fig,'Figure3')



def figure4(bins):
    from performance_phase_figure_20260916 import build
    build()

def supplement1(bins):
    fig,axs=plt.subplots(2,2,figsize=(9,6.5));fig.subplots_adjust(hspace=.6,wspace=.55)
    for ax,letter,label,g in [(axs[0,0],'A','AVP-Bmal1 group',bins[bins.Genotype=='AB']),(axs[0,1],'B','VIP-Bmal1 group',bins[bins.Genotype=='VB']),
        (axs[1,0],'C','Glucose',bins[bins.fluid=='glucose']),(axs[1,1],'D','Water A / total',bins[bins.fluid=='water'])]:
        panel(ax,letter,f'{label} (n={g.MouseID.nunique()})');individual_lines(ax,g,'total_g',BLUE);profile(ax,g,'total_g',BLUE,fit=True);twin=ax.twinx();individual_lines(twin,g,'preference',ORANGE,.12);profile(twin,g,'preference',ORANGE,fit=True);ax.tick_params(axis='y',colors=BLUE);twin.tick_params(axis='y',colors=ORANGE)
        ax.text(.02,.92,'; '.join(sorted(g.Condition.unique())),transform=ax.transAxes,fontsize=6)
    fig.text(.5,.006,'Genotype panels pool source-labeled LD/DD/saccharin conditions. These exploratory profiles do not establish absence of a clock contribution.',ha='center',fontsize=7)
    save(fig,'FigureS1')

def supplement2(bins):
    fig,axs=plt.subplots(2,2,figsize=(9,7));fig.subplots_adjust(hspace=.65,wspace=.4)
    A=np.array([38.4842,38.1546,38.0838,37.7695,37.9597,37.7434,37.9936,37.9034,38.2161,38.0199]);B=np.array([37.6447,37.7790,37.5484,37.6745,37.6028,38.2678,38.2853,37.6207,37.8422,38.0286])
    t,p=stats.ttest_ind(A,B);save_table(pd.DataFrame([dict(n_A=10,n_B=10,t=t,p=p,source='combined/bottle_weight.ipynb cell 0; hand-entered arrays')]),'baseline_bottle_test')
    ax=axs[0,0];panel(ax,'A','Archived bottle-weight sample')
    for i,v in enumerate([A,B]):ax.scatter(np.full(10,i)+rng_for(str(i)).normal(0,.04,10),v,s=20,c=BLUE);ax.errorbar(i,v.mean(),stats.sem(v),fmt='o',c='k',capsize=5)
    ax.set_xticks([0,1],['Bottle 1','Bottle 2']);ax.set_ylabel('Weight (g)');ax.text(.04,.9,f'Independent t={t:.2f}, p={p:.3g}',transform=ax.transAxes)
    ax=axs[0,1];panel(ax,'B','Age-matched total intake');pairs=read('age_matched_intake_pairs')
    for i,sex in enumerate(['F','M']):
        g=pairs[pairs.Sex==sex]
        for _,r in g.iterrows():ax.plot([i*3,i*3+1],[r.water_total_g,r.palatable_total_g],c=PINK if sex=='F' else GRAY,alpha=.35,lw=.6)
        ax.scatter([i*3,i*3+1],[g.water_total_g.mean(),g.palatable_total_g.mean()],c=PINK if sex=='F' else GRAY,s=35)
    ax.set_xticks([0,1,3,4],['F water','F palatable','M water','M palatable'],rotation=20);ax.set_ylabel('Session total intake (g)')
    gr=read('group_cosinor').set_index('group').reindex([f'{sex} {age}' for sex in ['F','M'] for age in AGE_ORDER]).reset_index();ax=axs[1,0];panel(ax,'C','Dominance overlap and permutation p')
    x=np.arange(len(gr));ax.scatter(x,gr.dominance_overlap,c=GRAY,label='Descriptive overlap');ax.scatter(x,gr.p_phase_permutation,c=BLUE,label='Permutation p');ax.axhline(.05,c='gray',ls='--');ax.set_xticks(x,gr.group,rotation=45,ha='right');ax.set_ylabel('Probability');ax.legend(frameon=False)
    ind=read('individual_cosinor');ax=axs[1,1];panel(ax,'D','Individual preference rhythms')
    for i,row in gr.iterrows():
        sex,age=row.group.split(' ');g=ind[ind.legacy_age_cohort & ind.metric.eq('preference') & ind.Sex.eq(sex)&ind.age_group.eq(age)]
        ax.bar(i-.15,100*np.mean(g.p<.05),.3,color=GRAY);ax.bar(i+.15,100*np.mean(g.q<.05),.3,color=BLUE)
    ax.set_xticks(x,gr.group,rotation=45,ha='right');ax.set_ylabel('% sessions');ax.plot([],[],color=GRAY,lw=5,label='Nominal p < .05');ax.plot([],[],color=BLUE,lw=5,label='BH q < .05');ax.legend(frameon=False)
    save(fig,'FigureS2')

def supplement3():
    fig,axs=plt.subplots(3,1,figsize=(WIDTH_IN,9),gridspec_kw={'height_ratios':[.8,1,1.45]});fig.subplots_adjust(hspace=.7,top=.94,bottom=.12)
    ax=axs[0];panel(ax,'A','Observed data and latent assumptions');ax.axis('off')
    labels=['Observed losses\nS, W per 3-h bin','Assumed events\nN × intake fraction','Predicted allocation\nExpected update','Observed allocation\nConditional score']
    for i,label in enumerate(labels):
        x=.02+i*.25;ax.text(x+.1,.5,label,transform=ax.transAxes,va='center',ha='center',fontsize=8,bbox={'boxstyle':'round,pad=.3','fc':'#f1f4f6','ec':'#adb7bf'})
        if i<3:ax.annotate('',xy=(x+.245,.5),xytext=(x+.215,.5),xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
    ax=axs[1];panel(ax,'B','Swap schedule follows elapsed session order')
    for i in range(8):ax.add_patch(Rectangle((i*3,0),3,1,facecolor=ORANGE if i%2==0 else BLUE,alpha=.8));ax.text(i*3+1.5,.5,'S | W' if i%2==0 else 'W | S',ha='center',va='center',color='white')
    ax.set_xlim(0,24);ax.set_ylim(0,1);ax.set_yticks([]);ax.set_xticks(np.arange(0,25,3));ax.set_xlabel('Elapsed time (h); initial side is arbitrary under symmetric initial values')
    ax=axs[2];panel(ax,'C','Model assumptions');ax.axis('off')
    rows=[['Random','0','Constant 0.5'],['Bias','1','Constant sweet preference'],['Linear / day-night','2','Non-learning drift or\nreference-phase step'],['Simple','3','Expected learning update; fixed reward\nand inverse temperature'],['Circadian reward','5','24-h reward sinusoid; shared phase'],['Full','6','Reward and inverse-temperature\nmodulation; shared phase']]
    table=ax.table(cellText=rows,colLabels=['Class','k','Mechanism'],colWidths=[.26,.08,.66],bbox=[0,-.25,1,1.2],cellLoc='left');table.auto_set_font_size(False);table.set_fontsize(8)
    for cell in table.get_celld().values():cell.set_edgecolor('#d0d0d0');cell.set_linewidth(.5)
    save(fig,'FigureS3')

def landscapes(bins):
    fits=read('corrected_expected_model_fits');best=fits[(fits.model=='simple')&fits.converged].sort_values(['cross_entropy','MouseID']);mid=best.iloc[len(best)//2].MouseID
    g=bins[bins.MouseID==mid].sort_values('Time_start');zt=g.ZT.to_numpy(float);el=g.elapsed_h.to_numpy(float);idx=g.bin_index.to_numpy(np.int64);n=allocate(g.total_g,1000);w=g.total_g.to_numpy()/g.total_g.sum();y=g.preference.to_numpy()
    manifest=[]
    for name,model in [('FigureS4','simple'),('FigureS5','reward'),('FigureS6','full')]:
        r=fits[(fits.MouseID==mid)&(fits.model==model)].iloc[0];par=np.array(json.loads(r.parameters));bounds=np.array(bounds_for(model));k=len(par)
        fig,axs=plt.subplots(k,k,figsize=(max(7,k*1.8),max(6.5,k*1.65)));fig.subplots_adjust(hspace=.4,wspace=.4,top=.87,bottom=.10)
        parameter_labels=[r'$\log_{10}\alpha$',r'$\log\beta_0$',r'$r_0$',r'$A_r$',r'$\phi$ (rad)',r'$A_\beta$']
        def loss(p):
            ph=np.clip(predict(model,p,zt,el,idx,n),1e-10,1-1e-10);ok=np.isfinite(y);return -np.sum(w[ok]*(y[ok]*np.log(ph[ok])+(1-y[ok])*np.log1p(-ph[ok])))
        base=loss(par)
        for i in range(k):
            for j in range(k):
                ax=axs[i,j]
                if j>i:ax.axis('off');continue
                if i==j:
                    x=np.linspace(*bounds[i],60);ys=[]
                    for v in x:p=par.copy();p[i]=v;ys.append(1000*(loss(p)-base))
                    ax.plot(x,ys,c=MODEL_COLOR[model]);ax.axvline(par[i],c='#b14750',ls='--',lw=.8)
                else:
                    x=np.linspace(*bounds[j],25);yy=np.linspace(*bounds[i],25);z=np.zeros((25,25))
                    for a,v in enumerate(yy):
                        for b,u in enumerate(x):p=par.copy();p[i]=v;p[j]=u;z[a,b]=1000*(loss(p)-base)
                    im=ax.imshow(z,origin='lower',extent=[*bounds[j],*bounds[i]],aspect='auto',cmap='viridis_r',vmin=0,vmax=100);ax.plot(par[j],par[i],'r*',ms=5)
                if i==k-1:ax.set_xlabel(parameter_labels[j])
                else:ax.tick_params(labelbottom=False)
                if j==0:ax.set_ylabel('Δ loss' if i==j else parameter_labels[i])
                elif i!=j:ax.tick_params(labelleft=False)
                elif i==j:
                    ax.set_title('Δ loss',fontsize=6,pad=2)
                    ax.yaxis.tick_right()
                ax.tick_params(labelsize=6)
        colorax=fig.add_axes([.43,.78,.16,.018] if model=='simple' else [.61,.82,.27,.015]);fig.colorbar(im,cax=colorax,orientation='horizontal',label='Δ loss (0–100 shown)')
        if model=='simple':
            ax=axs[0,2];ax.axis('on');bp=np.array(json.loads(fits[(fits.MouseID==mid)&fits.model.eq('bias')].iloc[0].parameters));xx=np.linspace(-4,4,160)
            def bias_loss(v):
                ph=predict('bias',np.array([v]),zt,el,idx,n);ok=np.isfinite(y);return -np.sum(w[ok]*(y[ok]*np.log(ph[ok])+(1-y[ok])*np.log1p(-ph[ok])))
            ax.plot(xx,[1000*(bias_loss(x)-bias_loss(bp[0])) for x in xx],c=GRAY);ax.axvline(bp[0],c='#b14750',ls='--',lw=.8);ax.set(xlabel='Bias intercept',ylabel='Δ loss',title='Bias model')
        fig.text(.5,.975,f'{LABEL[model]}'+(' + Bias' if model=='simple' else '')+': conditional loss slices',fontsize=10,va='top',ha='center')
        short_id=mid.split('_')[0];file_date=mid.split('_')[1][:8];fig.text(.5,.943,f'Same representative animal: {short_id}; session {file_date}',fontsize=8,ha='center')
        fig.text(.5,.012,'Δ negative log-score, scaled to 1,000 assumed trials. Other parameters fixed; slices do not prove identifiability or global optimality.',ha='center',fontsize=6.5)
        save(fig,name);manifest.append(dict(figure=name,MouseID=mid,model=model,selection='Median cross-entropy among converged Simple fits, same session for all landscapes'))
    save_table(pd.DataFrame(manifest),'landscape_selection')

def reviewer_supplement():
    fig,axs=plt.subplots(3,2,figsize=(WIDTH_IN,11.5));axs=axs.reshape(2,3);fig.subplots_adjust(left=.14,right=.96,top=.94,bottom=.09,hspace=.70,wspace=.55)
    ax=axs[0,0];panel(ax,'A','Constant-bias null')
    prof=read('constant_bias_null_profiles');prof=prof[prof.N_assumed==1000]
    for schedule,color in [('observed',BLUE),('uniform',GRAY),('shuffled',ORANGE)]:
        g=prof[prof.schedule==schedule];ax.plot(g.ZT,g['mean'],'o-',c=color,label=schedule);ax.fill_between(g.ZT,g.low,g.high,color=color,alpha=.12)
    ax.set_xlabel('Source ZT/CT (h)');ax.set_ylabel('Population preference');ax.legend(frameon=False)
    ax=axs[0,1];panel(ax,'B','Intake opportunity counterfactual')
    c=read('trial_availability_counterfactual');models=['bias','simple','reward','full']
    for i,s in enumerate(['observed','uniform','shuffled']):
        v=c[c.schedule==s].set_index('model').loc[models,'expected_night_minus_day'];ax.bar(np.arange(4)+(i-1)*.23,v,.23,label=s)
    ax.axhline(0,c='gray',lw=.7);ax.set_xticks(range(4),[LABEL[m] for m in models],rotation=35,ha='right');ax.set_ylabel('Expected night − day preference');ax.legend(frameon=False,fontsize=8)
    ax=axs[0,2];panel(ax,'C','Held-out contiguous bins')
    cv=read('expected_model_blocked_cv').groupby(['MouseID','model']).heldout_weighted_mse.mean().unstack();means=cv.mean().sort_values();ax.bar(range(len(means)),means,yerr=cv.sem().reindex(means.index),color=[MODEL_COLOR[m] for m in means.index],capsize=2);ax.set_xticks(range(len(means)),[LABEL[m] for m in means.index],rotation=30,ha='right');ax.set_ylabel('Conditional held-out weighted MSE')
    ax=axs[1,0];panel(ax,'D','Sensitivity to arbitrary score scale');s=read('pseudo_count_penalty_sensitivity')
    for model,g in s.groupby('model'):ax.plot(g.pseudo_count,g.n,'o-',c=MODEL_COLOR[model],label=LABEL[model])
    ax.set_xscale('log');ax.set_ylim(0,read('corrected_expected_model_fits').MouseID.nunique());ax.set_xlabel('Assumed count used in score penalty');ax.set_ylabel('Winning animals');ax.legend(ncol=2,frameon=False,fontsize=5.5,loc='upper center')
    ax=axs[1,1];panel(ax,'E','Model recovery: 20 replicates each');r=read('model_recovery');order=['bias','linear_bias','simple','reward','full'];ct=pd.crosstab(r.true_model,r.selected_model).reindex(index=order,columns=order,fill_value=0)/20
    im=ax.imshow(ct,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(5),[LABEL[m] for m in order],rotation=35,ha='right');ax.set_yticks(range(5),[LABEL[m] for m in order]);ax.set_xlabel('Selected');ax.set_ylabel('Generating class')
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{ct.iloc[i,j]:.0%}',ha='center',va='center',color='white' if ct.iloc[i,j]>.5 else 'black',fontsize=7)
    ax=axs[1,2];panel(ax,'F','Denominator sensitivity');s=read('low_intake_sensitivity');ax.plot(s.min_total_g,s.n_nominal_sig,'o-',color=ORANGE,label='Nominal');ax.plot(s.min_total_g,s.n_FDR_sig,'o-',c=BLUE,label='FDR');ax.set_xlabel('Minimum bin intake (g)');ax.set_ylabel('Correlated sessions');ax.legend(frameon=False)
    save(fig,'FigureS7_reviewer_checks')

def trial_sensitivity_figure():
    fig,axs=plt.subplots(2,2,figsize=(11,8));fig.subplots_adjust(hspace=.65,wspace=.4)
    ax=axs[0,0];panel(ax,'A','Score scale only; fixed learning events')
    data=read('pseudo_count_penalty_sensitivity');n=read('corrected_expected_model_fits').MouseID.nunique()
    for model,g in data.groupby('model'):ax.plot(g.pseudo_count,g.n,'o-',c=MODEL_COLOR[model],label=LABEL[model],ms=3)
    ax.set(xscale='log',xlabel='Assumed score count N',ylabel=f'Winning animals (of {n})',ylim=(0,n));ax.legend(ncol=2,frameon=False,fontsize=6)
    sample=read('latent_count_shape_refit_sensitivity');sine=sample[sample['shape'].eq('sine')];ids=sorted(sine.MouseID.unique());order=['bias','linear_bias','simple','reward','full']
    winners=sine.loc[sine.groupby(['MouseID','N_assumed']).pseudo_AIC.idxmin()]
    ax=axs[0,1];panel(ax,'B',f'Learning-event count refitted ({len(ids)} animals)')
    counts=pd.crosstab(winners.N_assumed,winners.model).reindex(columns=order,fill_value=0)
    for model in order:ax.plot(counts.index,counts[model],'o-',c=MODEL_COLOR[model],label=LABEL[model],ms=3)
    ax.set(xscale='log',xlabel='Assumed learning events and score count N',ylabel='Winning animals',ylim=(0,len(ids)+.7));ax.legend(ncol=2,frameon=False,fontsize=6)
    ax=axs[1,0];panel(ax,'C','Refitted error on the same interval data')
    for model in order:
        g=sine[sine.model.eq(model)].groupby('N_assumed').cross_entropy.agg(['mean','sem']);ax.errorbar(g.index,g['mean'],yerr=g['sem'],c=MODEL_COLOR[model],fmt='o-',capsize=3,ms=3,label=LABEL[model])
    ax.set(xscale='log',xlabel='Assumed learning events N',ylabel='Intake-weighted cross-entropy');ax.legend(ncol=2,frameon=False,fontsize=6)
    ax=axs[1,1];panel(ax,'D','Count and reward-shape sensitivity')
    rows=[]
    for shape in ['sine','square']:
        for N in [100,1000,10000]:
            g=sample[sample['shape'].eq(shape)&sample.N_assumed.eq(N)];best=g.loc[g.groupby('MouseID').pseudo_AIC.idxmin()].set_index('MouseID').model
            rows.append([order.index(best.loc[mid]) for mid in ids])
    from matplotlib.colors import ListedColormap
    ax.imshow(rows,aspect='auto',cmap=ListedColormap([MODEL_COLOR[m] for m in order]),vmin=-.5,vmax=4.5)
    ax.set_yticks(range(6),[f'{shape}, {N}' for shape in ['Sine','Sq.'] for N in ['100','1k','10k']]);ax.set_xticks(range(len(ids)),range(1,len(ids)+1));ax.set_xlabel('Same selected animals across all refits')
    save_table(pd.DataFrame(dict(plot_index=np.arange(1,len(ids)+1),MouseID=ids)),'figureS8_animal_order')
    fig.text(.5,.01,'A uses all animals and ten candidates; B–D use the same deterministic 12-animal subset and five candidates. Assumed trials are not observed licks.',ha='center',fontsize=7)
    save(fig,'FigureS8_trial_sensitivity')

