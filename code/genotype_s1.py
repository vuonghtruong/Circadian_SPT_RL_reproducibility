"""S1 profiles plus direct genotype phase comparisons using Figure 2's grammar."""
import json
import numpy as np
from matplotlib.lines import Line2D

COLORS={'Control':'#e89127','AB':'#c54b98','VB':'#288e83'}
MARKERS={'Control':'o','AB':'s','VB':'^'}

def install(f,renderer):
    renderer['HEIGHT_MM']['FigureS1']=170
    def supplement1(bins):
        fig=f.plt.figure(figsize=(175/25.4,170/25.4))
        phase=f.read('genotype_phase_inclusions')
        eligible=bins[bins.MouseID.isin(phase.MouseID)]
        groups=[('AB',eligible[eligible.Genotype.eq('AB')]),('VB',eligible[eligible.Genotype.eq('VB')]),
                ('Glucose',bins[bins.fluid.eq('glucose')]),('Water',bins[bins.fluid.eq('water')])]
        for index,(title,g) in enumerate(groups):
            row,column=divmod(index,2)
            ax=fig.add_axes([.13+.51*column,.79-.27*row,.285,.14])
            f.panel(ax,'ABCD'[index],title)
            f.individual_lines(ax,g,'total_g',f.BLUE,.15)
            f.profile(ax,g,'total_g',f.BLUE,fit=True)
            ax.set(xlim=(0,24),ylim=(0,5.5),xticks=[0,6,12,18,24],yticks=[0,2,4],xlabel='ZT/CT (h)',ylabel='Total intake (g)')
            ax.tick_params(axis='y',colors=f.BLUE);ax.yaxis.label.set_color(f.BLUE)
            if row==0:ax.set_xlabel('')
            twin=ax.twinx();f.individual_lines(twin,g,'preference',f.ORANGE,.15)
            f.profile(twin,g,'preference',f.ORANGE,fit=True)
            twin.set(ylim=(0,1),yticks=[0,.5,1],ylabel='Preference',xlabel='')
            twin.tick_params(axis='y',colors=f.ORANGE);twin.yaxis.label.set_color(f.ORANGE)
            twin.spines['right'].set_visible(True)

        tests=f.read('genotype_vs_control_phase_tests')
        handles=[Line2D([],[],ls='',marker=MARKERS[g],color=COLORS[g],markersize=5,
                 label='Control genotype' if g=='Control' else g) for g in COLORS]
        fig.legend(handles=handles,loc='center',bbox_to_anchor=(.52,.415),ncol=3,frameon=False,
                   handletextpad=.5,columnspacing=1.7)
        fig.text(.52,.382,'LD + early DD pooled; n (Control / AB / VB) = 44 / 17 / 22',ha='center',va='center',fontsize=8)
        audit=[]
        for row,lighting in enumerate(['LD+DD']):
            for column,(metric,name) in enumerate([('total_phase','Total intake'),('pref_phase','Preference')]):
                ax=fig.add_axes([.12+.51*column,.04,.305,.28],projection='polar')
                letter='EF'[column];f.panel(ax,letter,name)
                ax.set_theta_zero_location('N');ax.set_theta_direction(-1)
                ax.set_ylim(0,1.35);ax.set_yticks([]);ax.set_xticks([]);ax.grid(False)
                for angle,label in zip(np.arange(4)*np.pi/2,['ZT/CT0','6','12','18']):
                    ax.text(angle,1.25,label,ha='center',va='center',fontsize=8)
                ax.spines['polar'].set_visible(False)
                for angle in np.arange(4)*np.pi/2:ax.plot([angle,angle],[0,1],c='#dddddd',lw=.7,zorder=0)
                theta=np.linspace(np.pi,2*np.pi,100);ax.fill_between(theta,0,1,color='#ededed',zorder=0)
                ax.plot(np.linspace(0,2*np.pi,300),np.ones(300),c='k',lw=1.1,zorder=1)
                counts=[];means={}
                for genotype in COLORS:
                    group=phase[phase.Genotype.eq(genotype)].dropna(subset=[metric])
                    values=group[metric].to_numpy(float)
                    z=np.exp(1j*values*np.pi/12).mean();mu=np.angle(z)%(2*np.pi);R=abs(z)
                    means[genotype]=mu;counts.append(len(values))
                    radii=f.rng_for('genotype'+lighting+metric+genotype).uniform(1.025,1.13,len(values))
                    ax.scatter(values*np.pi/12,radii,c=COLORS[genotype],marker=MARKERS[genotype],s=12,alpha=.65,lw=0,clip_on=False,zorder=2)
                    ax.plot([mu,mu],[0,R],color=COLORS[genotype],lw=1.5,zorder=3)
                    ax.scatter(mu,R,s=48,c=COLORS[genotype],marker=MARKERS[genotype],edgecolors='white',lw=.6,zorder=5)
                    audit.append(dict(panel=letter,lighting=lighting,metric=metric,genotype=genotype,n=len(values),mean_phase_h=mu*12/np.pi,R=R,individuals_unclipped=True))
                for j,genotype in enumerate(['AB','VB']):
                    test=tests[tests.metric.eq(metric)&tests.genotype.eq(genotype)].iloc[0]
                    delta=(means[genotype]-means['Control']+np.pi)%(2*np.pi)-np.pi
                    radius=.53+.18*j
                    theta=means['Control']+np.linspace(0,delta,100)
                    ax.plot(theta,np.full(len(theta),radius),c=COLORS[genotype],lw=1.25,zorder=4)
                    for endpoint in [theta[0],theta[-1]]:ax.plot([endpoint,endpoint],[radius-.025,radius+.025],c=COLORS[genotype],lw=1.25,zorder=4)
                    label=f.stars(test.q_BH).replace('n.s.','ns')
                    ax.text(means['Control']+delta/2,radius+.075,label,color=COLORS[genotype],ha='center',va='center',fontsize=8,zorder=6)
        renderer['GENOTYPE_S1_QA']=audit
        f.save(fig,'FigureS1')
        (renderer['DEST']/'audit/genotype_s1_phase_vectors.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    f.supplement1=supplement1
