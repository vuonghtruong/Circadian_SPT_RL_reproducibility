"""Read-only schedule audit and explicitly post-hoc phenotype exclusions.

Never infer a lighting correction from an animal's outcome. Primary data remain
unchanged unless independent protocol evidence establishes a correction.
"""
from common import *
import json,re,zipfile,xml.etree.ElementTree as ET
from statistics import cosinor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image





def make_figure(primary,bins,counts,cv):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    # Explicit placements leave room for source-batch labels at 175 mm width.
    fig=plt.figure(figsize=(175/25.4,10.2))
    axs=np.array([[fig.add_axes([.12,.55,.30,.34]),fig.add_axes([.72,.55,.25,.34])],
                  [fig.add_axes([.28,.13,.27,.32]),fig.add_axes([.72,.13,.25,.32])]])
    for schedule,g in primary.groupby('schedule'):
        label='LD/reference' if schedule=='standard LD/reference' else schedule
        axs[0,0].scatter(g.night_g,g.day_g,s=23,alpha=.75,label=f'{label} (n={len(g)})')
    lim=max(primary.day_g.max(),primary.night_g.max())*1.05
    axs[0,0].plot([0,lim],[0,lim],'k--',lw=1);axs[0,0].set(xlabel='Night-bin intake (g)',ylabel='Day-bin intake (g)',title='A  Recorded day/night intake');axs[0,0].legend(fontsize=8,loc='upper left',handletextpad=.3)
    order=primary.groupby('File').day_fraction.mean().sort_values().index
    for j,file in enumerate(order):
        g=primary[primary.File==file];rg=rng_for(file)
        axs[0,1].scatter(g.day_fraction,np.full(len(g),j)+rg.uniform(-.18,.18,len(g)),s=18,c=np.where(g.day_dominant,'#c65b46','#697580'),alpha=.8)
    labels=[re.search(r'\d{8}',x)[0]+' '+x.split('test')[-1].replace('.xlsx','').strip() for x in order]
    axs[0,1].set(yticks=np.arange(len(order)),yticklabels=labels,xlabel='Recorded day fraction',title='B  Day fraction by batch');axs[0,1].tick_params(axis='y',labelsize=8)
    axs[0,1].axvline(.5,c='k',ls='--',lw=1);axs[0,1].axvline(.6,c='k',ls=':',lw=1)
    scenarios=list(counts.analysis.unique());short=['All eligible','Exclude LD\nday > night','Exclude any\nday > night','Exclude day\nfraction ≥60%','Exclude unresolved\nschedules']
    base=np.zeros(len(scenarios));colors=plt.cm.Greys(np.linspace(.2,.9,8))
    for c,m in zip(colors,counts.model.unique()):
        y=counts[counts.model==m].set_index('analysis').loc[scenarios,'n'].to_numpy()
        axs[1,0].barh(np.arange(len(y)),y,left=base,label=m.replace('_no_swap',' NS').capitalize(),color=c,edgecolor='white',lw=.5);base+=y
    axs[1,0].set(yticks=range(5),yticklabels=short,xlabel='Animals',title='C  Model winners')
    handles,labels=axs[1,0].get_legend_handles_labels();fig.legend(handles,labels,fontsize=8,ncol=4,loc='lower center',bbox_to_anchor=(.52,.025),frameon=False,handlelength=1.3,columnspacing=1.)
    for m,color in [('bias','#333333'),('full','#b04b36')]:
        g=cv[cv.model==m].set_index('analysis').loc[scenarios]
        axs[1,1].errorbar(g.mean_mse,np.arange(5)+(-.07 if m=='bias' else .07),xerr=g['sem'],fmt='o',capsize=3,label=m,color=color)
    axs[1,1].set(yticks=range(5),yticklabels=['']*5,xlabel='Conditional held-out MSE',title='D  Held-out error');axs[1,1].legend(loc='upper center',handletextpad=.3)
    fig.suptitle('Light–dark intake audit and post-hoc exclusion sensitivity',fontsize=14)
    from figures import save
    save(fig,'FigureS9_light_dark_audit')
    flagged=primary[primary.day_dominant].sort_values('day_fraction',ascending=False)
    pages=[];page_ids=[]
    for start in range(0,len(flagged),9):
        fig,axs=plt.subplots(3,3,figsize=(11,9),layout=None)
        fig.subplots_adjust(left=.075,right=.985,bottom=.075,top=.90,wspace=.32,hspace=.50)
        for ax,(_,row) in zip(axs.flat,flagged.iloc[start:start+9].iterrows()):
            g=bins[bins.MouseID==row.MouseID].sort_values('ZT')
            ax.axvspan(12,24,color='.88');ax.plot(g.ZT,g.total_g,'o-',c='#333333',ms=3);ax.plot(g.ZT,g.water_g,':',c='#527eab',label='Water');ax.plot(g.ZT,g.reward_g,':',c='#d1843c',label='Sweet')
            date=re.search(r'\d{8}',row.File)[0]
            ax.set(title=f'{date} {row.Mouse} | {row.Condition}\nDay fraction={row.day_fraction:.3f}; R²={row.total_cosinor_r2:.2f}',xlabel='Recorded ZT / reference time (h)',ylabel='Intake / 3 h (g)',xticks=[0,6,12,18,24],xlim=(0,24));ax.legend(fontsize=7);ax.title.set_fontsize(8);ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8);ax.tick_params(labelsize=7)
        for ax in list(axs.flat)[len(flagged.iloc[start:start+9]):]:ax.set_visible(False)
        fig.suptitle('Flagged profiles: shading is the recorded night/reference half, not verified lighting',fontsize=11)
        # The diagnostic PDF embeds 250-dpi Matplotlib pages to avoid a PDF font
        # rendering defect. The thirteen manuscript figures remain vector PDFs.
        pagefile=OUT/'audit/pdf_qa'/f'flagged_profiles_page{start//9+1}.png'
        pagefile.parent.mkdir(exist_ok=True,parents=True)
        fig.savefig(pagefile,dpi=250);plt.close(fig)
        with Image.open(pagefile) as page:pages.append(page.convert('RGB'))
        page_ids.append(flagged.iloc[start:start+9].MouseID.tolist())
    pages[0].save(OUT/'audit/light_dark_flagged_profiles.pdf',save_all=True,append_images=pages[1:],resolution=250,quality=98)
    for page in pages:page.close()
    (OUT/'audit/light_dark_profile_pages.json').write_text(json.dumps(page_ids,indent=2))


