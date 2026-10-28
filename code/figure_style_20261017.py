"""Render publication figures from frozen 2026-09-16 tables; never refit data.

Run from this package: python reproduce.py
Portable use: python code/figure_style_20261017.py --base . --dest reproduced
The baseline scripts/tables and existing outputs are read-only inputs.
"""
from pathlib import Path
import ast, hashlib, inspect, json, re, sys, textwrap
import argparse
SCRIPT_DIR = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',type=Path,default=SCRIPT_DIR.parent)
parser.add_argument('--dest',type=Path,default=SCRIPT_DIR.parent)
args=parser.parse_args()
# Local copies take precedence in a portable sharing bundle.
import common

BASE = args.base.resolve()
DEST = args.dest.resolve()
common.OUT = BASE
import figures as f
import publication_style as old_style
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image
import pypdfium2 as pdfium

HEIGHT_MM = {'Figure1':195,'Figure2':195,'Figure3':205,'Figure4':190,
 'FigureS1':165,'FigureS2':185,'FigureS3':185,'FigureS4':180,'FigureS5':195,
 'FigureS6':205,'FigureS7_reviewer_checks':210,'FigureS8_trial_sensitivity':185,
 'FigureS9_light_dark_audit':210,'FigureS10_model_phases':210,
 'FigureS11_model_phase_examples':210}
MANIFEST=[]
FOOTNOTES={}
PATCHED_SOURCES={}
INPUTS=set()
POLAR_QA=[]
_read_csv=f.pd.read_csv
def logged_read_csv(path,*args,**kwargs):
    if isinstance(path,(str,Path)):
        item=Path(path).resolve()
        if item.is_file():INPUTS.add(str(item))
    return _read_csv(path,*args,**kwargs)
f.pd.read_csv=logged_read_csv

def compact_finalize(fig,name):
    """Apply purely graphical changes, preserving all numerical coordinates."""
    for ax in fig.axes:
        for spine in ax.spines.values():spine.set_linewidth(1.1)
        ax.tick_params(which='major',width=1.1,length=3,pad=2)
        if ax.name=='polar':
            ax.tick_params(axis='x',length=0,pad=3)
            if name=='Figure1':
                # The former r_max=1.22 placed the r=1 mouse phases inside the
                # outline. Keep the phase data and resampled R unchanged and
                # make the circular outline represent R=1 exactly.
                ax.set_ylim(0,1)
                ax.set_yticks([.5,1]);ax.set_yticklabels([])
                ax.spines['polar'].set_zorder(0)
                for collection in ax.collections:
                    offsets=collection.get_offsets()
                    if len(offsets) and f.np.allclose(offsets[:,1],1):
                        collection.set_clip_on(False);collection.set_zorder(5)
                markers=[c for c in ax.collections if len(c.get_offsets()) and f.np.allclose(c.get_offsets()[:,1],1)]
                POLAR_QA.append({'outline_radius':float(ax.get_ylim()[1]),'individual_phase_radius':1.0,'individual_points':sum(len(c.get_offsets()) for c in markers),'all_unclipped':all(not c.get_clip_on() for c in markers)})
        if name=='FigureS11_model_phase_examples' and ax.name!='polar':
            ax.set_xticks([0,12,24])
            if ax.get_xlabel():ax.set_xlabel('Reference time (h)')
        if name=='FigureS10_model_phases' and ax.name!='polar' and not hasattr(ax,'_colorbar'):
            if tuple(ax.get_xlim())==(0.0,24.0):ax.set_xticks([0,12,24])
            if tuple(ax.get_ylim())==(0.0,24.0):ax.set_yticks([0,12,24])
        if name=='Figure2':
            from matplotlib.transforms import ScaledTranslation
            for text_ in ax.texts:
                if getattr(text_,'_panel_letter',False) and text_.get_text() in ['G','H']:
                    text_.set_transform(ax.transAxes+ScaledTranslation(-18/72,3/72,fig.dpi_scale_trans))

def install_compact_style():
    """Shorter labels and revised panel geometry for the A4 manuscript."""
    f.BLUE='#3478a3';f.ORANGE='#cc852f';f.PINK='#b8679f';f.GREEN='#4f9488'
    f.MODEL_COLOR.update({'simple':f.BLUE,'simple_no_swap':'#86b0c7',
                         'reward':f.ORANGE,'reward_no_swap':'#d9ad76',
                         'full':'#aa535b','full_no_swap':'#cc9399'})
    replace_function(f,'figure1',[
        ("bottom=.075,top=.94,hspace=.77,wspace=.43", "bottom=.08,top=.935,hspace=.82,wspace=.50"),
        ("'Observed R / mean null R'", "'Observed / null R'"),
        ("ax.set_ylim(.001,1);ax.legend(frameon=False)", "ax.set_ylim(.001,1);ax.legend(frameon=False,ncol=2,loc='upper left',handlelength=.8,handletextpad=.3,columnspacing=.6)"),
        ("f'{sex}, {a.replace(\"EarlyMid\",\"Early-mid\")} (n={g.MouseID.nunique()})'", "f'{sex}, {a.replace(\"EarlyMid\",\"Early-mid\")}\\n(n={g.MouseID.nunique()})'")])
    replace_function(f,'figure2',[
        ("top=.955,bottom=.625,hspace=.43,wspace=.32", "top=.955,bottom=.625,hspace=.55,wspace=.32"),
        ("hspace=.20", "hspace=.27"),
        ("[.085,.425,.14,.095]", "[.085,.42,.14,.095]"),
        ("[.305,.425 if letter=='D' else .265,.14,.105]", "[.305,.42 if letter=='D' else .26,.14,.105]"),
        ("[.085,.265,.14,.105]", "[.085,.26,.14,.105]"),
        ("bbox_to_anchor=(.35,.568)", "bbox_to_anchor=(.35,.57)"),
        ("bbox_to_anchor=(.746,.557)", "bbox_to_anchor=(.746,.565)"),
        ("[.45+.295*i,.048,.225,.18]", "[.45+.295*i,.065,.225,.165]"),
        ("ax.text(.5,-.18", "ax.text(.5,-.20"),
        ("[.535+.25*i,.335,.195,.195]", "[.535+.25*i,.325,.195,.195]"),
        ("'Total intake\\nacrophase (h)'", "'Intake acrophase (h)'"),
        ("[.085,.045,.29,.145]", "[.085,.035,.29,.16]"),
        ("'RF-morning':'#63c849','RF-afternoon':'#269b45'", "'RF-morning':'#77aa50','RF-afternoon':'#397c4e'"),
        ("'Control':'#ed9418','PA':'#ef5558'", "'Control':'#dba03f','PA':'#c96565'")])
    replace_function(f,'figure3',[
        ("hspace=1.00,wspace=1.2", "hspace=1.15,wspace=1.2"),
        ("height_ratios=[1,1,1,1.2,1.25,1.25]", "height_ratios=[1.1,.75,.75,1.2,1.25,1.25]"),
        ("rotation=60", "rotation=45"),
        ("'Mean delta pseudo-AIC'", "'Mean score difference'"),
        ("'Animal ordered by best-fitting class'", "'Animal order'"),
        ("'Delta score; capped at 100'", "'Score difference'"),
        ("'Score differences across animals'", "'Score differences'"),
        ("rotation=30", "rotation=35")])
    replace_function(f,'supplement1',[("hspace=.6,wspace=.55", "left=.11,right=.89,bottom=.12,top=.91,hspace=.62,wspace=.7")])
    replace_function(f,'supplement2',[("hspace=.65,wspace=.4", "left=.12,right=.96,bottom=.16,top=.92,hspace=.68,wspace=.62"),("ax.set_ylabel('% sessions');", "ax.set_ylabel('% sessions');ax.set_ylim(0,14);")])
    replace_function(f,'supplement3',[("hspace=.7,top=.94,bottom=.12", "hspace=.65,top=.92,bottom=.13")])
    replace_function(f,'reviewer_supplement',[
        ("top=.94,bottom=.09,hspace=.70,wspace=.55", "top=.93,bottom=.10,hspace=.85,wspace=.68"),
        ("'Intake opportunity counterfactual'", "'Intake opportunity'"),
        ("'Sensitivity to arbitrary score scale'", "'Score-scale sensitivity'"),
        ("'Model recovery: 20 replicates each'", "'Model recovery (20 / class)'"),
        ("'Expected night − day preference'", "'Night − day preference'"),
        ("'Conditional held-out weighted MSE'", "'Held-out weighted MSE'"),
        ("'Assumed count used in score penalty'", "'Assumed score count'"),
        ("ax.legend(ncol=2,frameon=False,fontsize=5.5,loc='upper center')", "ax.legend(ncol=2,frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.03),handlelength=.8,handletextpad=.3,columnspacing=.5,labelspacing=.15)")])
    replace_function(f,'trial_sensitivity_figure',[
        ("hspace=.65,wspace=.4", "left=.13,right=.96,bottom=.12,top=.93,hspace=.66,wspace=.6"),
        ("'Score scale only; fixed learning events'", "'Score scale; fixed events'"),
        ("f'Learning-event count refitted ({len(ids)} animals)'", "f'Event-count refits (n={len(ids)})'"),
        ("'Refitted error on the same interval data'", "'Refitted interval error'"),
        ("'Assumed learning events and score count N'", "'Assumed events / score count N'"),
        ("'Same selected animals across all refits'", "'Animal order (same subset)'")])
    import performance_phase_figure_20260916 as pf
    replace_function(pf,'build',[("hspace=.70,wspace=.68", "hspace=.77,wspace=.72")])
    from reference_style import install
    install(f, globals())
    from likelihood_clarity import install as likelihood_install
    from phase_clarity import install as phase_install
    from supplement_clarity import install as supplement_install
    likelihood_install(f, globals())
    phase_install(f, globals())
    supplement_install(f, globals())
    from submission_figures import install as submission_install
    submission_install(f, globals())
    from genotype_s1 import install as genotype_install
    genotype_install(f, globals())
    replace_function(f,'reviewer_supplement',[
        ("'Held-out contiguous bins'", "'Omitted measurements'"),
        ("'Model recovery (20 / class)'", "'Model recovery'"),
        ("'Denominator sensitivity'", "'Low-intake sensitivity'")])

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def baseline_hashes():
    return {str(p.relative_to(BASE)):digest(p) for p in BASE.rglob('*')
            if p.is_file() and any(x in p.relative_to(BASE).parts[:1] for x in ['data','results','tables'])}

def no_table_write(table,name):
    """Plot order/statistic tables already exist and must remain unchanged."""
    assert (BASE/'results'/f'{name}.csv').exists(), name
    INPUTS.add(str((BASE/'results'/f'{name}.csv').resolve()))

def prepare(fig,name):
    fig.set_size_inches(175/25.4,HEIGHT_MM[name]/25.4,forward=True)
    # Detailed explanatory text belongs in the manuscript legends. Preserve a
    # separate ledger so that removing figure footers cannot lose disclosures.
    FOOTNOTES[name]=[t.get_text() for t in fig.texts if t.get_position()[1]<.04]
    for t in list(fig.texts):
        if t.get_position()[1]<.04:t.remove()
    for t in fig.findobj(match=Text):t.set_fontfamily('Arial');t.set_fontsize(8)
    for ax in fig.axes:
        width=175/25.4*72*ax.get_position().width
        title=ax.get_title(loc='left') or ax.get_title()
        if ax.get_title(loc='left'):ax.set_title('',loc='left')
        if title:ax.set_title(old_style._wrap(title,max(13,int(width/4.0))),loc='center',fontsize=9,pad=7)
        ax.xaxis.label.set_text(old_style._wrap(ax.xaxis.label.get_text(),max(13,int(width/3.8))))
        for t in ax.texts:
            if getattr(t,'_panel_letter',False):
                t.set_fontsize(11)
                if name=='Figure3':
                    from matplotlib.transforms import ScaledTranslation
                    t.set_transform(ax.transAxes+ScaledTranslation(-18/72,3/72,fig.dpi_scale_trans))
                if name=='Figure2' and t.get_text() in ['I','J']:
                    t.set_transform(ax.transAxes);t.set_position((.07,1.10))
            elif t.get_transform()==ax.transAxes and len(t.get_text())>40:
                t.set_text(old_style._wrap(t.get_text(),max(15,int(width/3.9))))
        if ax.name!='polar' and name in {'FigureS4','FigureS5','FigureS6'} and not hasattr(ax,'_colorbar'):
            ax.xaxis.set_major_locator(MaxNLocator(2));ax.yaxis.set_major_locator(MaxNLocator(2))
        if ax.get_legend():
            for t in ax.get_legend().get_texts():t.set_fontfamily('Arial');t.set_fontsize(8)
    for t in fig.texts:
        if t is getattr(fig,'_suptitle',None):
            t.set_fontsize(10);t.set_text(old_style._wrap(t.get_text(),85))
    compact_finalize(fig,name)
    fig.canvas.draw()

def save(fig,name):
    prepare(fig,name)
    output=DEST/'figures';output.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(output/f'{name}.{ext}',dpi=300,bbox_inches=None,
                    metadata={'Creator':'Circadian SPT verified-table figure renderer, 2026-10-17'} if ext=='pdf' else None)
    if name in ['Figure1','Figure2','Figure3','Figure4']:
        # RGB LZW TIFF at 600 dpi satisfies the official PLOS image format.
        import io
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=600,bbox_inches=None)
        buf.seek(0)
        with Image.open(buf) as im:im.convert('RGB').save(output/(name.replace('Figure','Fig')+'.tif'),compression='tiff_lzw',dpi=(600,600))
    f.plt.close(fig)
    print(name,flush=True)

def replace_function(module,name,replacements):
    key=(module.__name__,name)
    source=PATCHED_SOURCES.get(key) or inspect.getsource(getattr(module,name))
    for old,new in replacements:
        assert old in source,(name,old[:80]);source=source.replace(old,new)
    PATCHED_SOURCES[key]=source
    exec(compile(source,str(Path(__file__))+'::'+name,'exec'),module.__dict__)

def install_style():
    f.save=save;f.save_table=no_table_write
    f.plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
      'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
      'axes.linewidth':1.1,'xtick.major.width':1.1,'ytick.major.width':1.1,'lines.linewidth':1,'savefig.facecolor':'white'})
    # Restore the submitted figure's sex/age color cues, using muted tones.
    f.PINK='#ca4ea7';f.AGE_COLOR={'Young':'#dd8a2c','EarlyMid':'#8d85ba','Mid':'#cd629d','Old':'#78a542'}
    replace_function(f,'figure1',[
      ("bottom=.085,top=.94,hspace=.8,wspace=.40","bottom=.075,top=.94,hspace=.77,wspace=.43"),
      ("ax.text(.5,.02,'BH FDR vs water',transform=ax.transAxes,ha='center',fontsize=8)",""),
      ("ax.set_ylabel('')\n            s=g.groupby", "ax.set_ylabel('');ax.set_yticks([0,.5,1]);ax.tick_params(labelleft=sex=='F')\n            s=g.groupby")])
    # Explicit compact placement mirrors the submitted C/F-D/E-G/H layout.
    replace_function(f,'figure2',[
      ("gs=fig.add_gridspec(5,4,height_ratios=[1.65,1.65,1.25,2.4,1.65],left=.10,right=.91,top=.955,bottom=.09,hspace=.95,wspace=.72)",
       "gs=fig.add_gridspec(2,4,left=.085,right=.93,top=.955,bottom=.625,hspace=.43,wspace=.32)"),
      ("cell=top[j].subgridspec(2,1,height_ratios=[1,.8],hspace=.12)","cell=top[j].subgridspec(2,1,height_ratios=[1,.65],hspace=.20)"),
      ("f'{a.replace(\"EarlyMid\",\"Early-mid\")}\\n{sex}, n={g.MouseID.nunique()}'","f'{a.replace(\"EarlyMid\",\"Early-mid\")} ({g.MouseID.nunique()})'"),
      ("ax.set_ylim(0,5);twin", "ax.set_ylim(0,5);ax.set_yticks([0,2,4]);twin"),
      ("if j:ax.tick_params(labelleft=False)","if j:ax.tick_params(labelleft=False)"),
      ("lower.legend(frameon=False,fontsize=8,loc='upper left',handlelength=1,handletextpad=.3,labelspacing=.15)","lower.legend(frameon=False,fontsize=8,loc='upper left',handlelength=.8,handletextpad=.2,labelspacing=.05,ncol=2,columnspacing=.4)"),
      ("if j:lower.tick_params(labelleft=False)", "if j:lower.tick_params(labelleft=False)\n            if row==0:lower.set_xlabel('')"),
      ("ax=fig.add_subplot(gs[2,0]);panel(ax,'C',f'Day/night (n={int(daytests.loc[\"total_g\",\"n\"])})')", "ax=fig.add_axes([.085,.425,.14,.095]);panel(ax,'C','Day / night')"),
      ("ax=fig.add_subplot(gs[2,column]);panel(ax,letter,'Intake phase (h)' if metric=='total_phase' else 'Preference phase (h)')", "ax=fig.add_axes([.305,.425 if letter=='D' else .265,.14,.105]);panel(ax,letter,'Intake phase' if metric=='total_phase' else 'Preference phase')"),
      ("if letter=='D':sex_handles,sex_labels=ax.get_legend_handles_labels()", "if letter=='D':sex_handles,sex_labels=ax.get_legend_handles_labels();ax.set_xlabel('');ax.tick_params(labelbottom=False)"),
      ("s=14,c=color,alpha=.55,label=sex", "s=8,c=color,alpha=.55,label=sex"),
      ("fig.legend(sex_handles,['Female','Male'],frameon=False,ncol=2,loc='center',bbox_to_anchor=(.515,gs[2,1:3].get_position(fig).y1+.044),handlelength=.7,columnspacing=1.)", "fig.legend(sex_handles,['Female','Male'],frameon=False,ncol=2,loc='center',bbox_to_anchor=(.35,.568),handlelength=.6,columnspacing=.8)"),
      ("ax=fig.add_subplot(gs[2,3]);panel(ax,'F','Bootstrap phase\\nconcentration')", "ax=fig.add_axes([.085,.265,.14,.105]);panel(ax,'F','Cross-corr.')"),
      ("ax=fig.add_subplot(gs[3,2*i:2*i+2]);panel(ax,'G' if i==0 else 'H','Female' if sex=='F' else 'Male')", "ax=fig.add_axes([.535+.25*i,.335,.195,.195]);panel(ax,'G' if i==0 else 'H','Female' if sex=='F' else 'Male')"),
      ("bbox_to_anchor=(.515,gs[3,:].get_position(fig).y1+.035),handlelength=.6,handletextpad=.4,ncol=4,columnspacing=.8", "bbox_to_anchor=(.746,.557),handlelength=.5,handletextpad=.3,ncol=2,columnspacing=.9"),
      ("bottom=gs[4,:].subgridspec(1,3,width_ratios=[1.15,1.1,1.1],wspace=.38);legend=fig.add_subplot(bottom[0])", "legend=fig.add_axes([.085,.045,.29,.145])"),
      ("legend.text(.5,1.05,'Entrainment condition',fontsize=9,ha='center')", "legend.text(.5,1.05,'Entrainment condition',fontsize=9,ha='center')"),
      ("legend.text(0,.09,'First session / condition.\\nAll pairs versus Control.\\nPA reference unverified.',fontsize=8,va='top')", "legend.text(0,.045,'First session / condition',fontsize=8,va='top')"),
      ("ax=fig.add_subplot(bottom[i+1],projection='polar')", "ax=fig.add_axes([.45+.295*i,.048,.225,.18],projection='polar')"),
      ("ax.set_ylim(0,1.27);ax.set_yticks([])", "ax.set_ylim(0,1.27);ax.tick_params(axis='x',pad=1);ax.set_yticks([])"),
      ("ax.text(.5,-.29,'; '.join(pair_labels)", "ax.text(.5,-.18,'; '.join(pair_labels)"),
      ("save(fig,'Figure2')", "fig.text(.012,.89,'Female',rotation=90,ha='center',va='center',color=PINK,fontsize=9)\n    fig.text(.012,.705,'Male',rotation=90,ha='center',va='center',color=GRAY,fontsize=9)\n    save(fig,'Figure2')")])
    replace_function(f,'figure3',[
      ("bottom=.105,hspace=.95,wspace=1.2", "bottom=.07,hspace=1.00,wspace=1.2"),
      ("ax.set_xticks(np.arange(0,25,3))", "ax.set_xticks([0,6,12,18,24])"),
      ("'Mean score difference'", "'Mean score difference'"),
      ("f'Winning count, best to worst (n={len(A)})'", "f'Model winners (n={len(A)})'")])
    # Boundary estimates remain at their exact parameter coordinates; show the
    # complete marker instead of cropping half the star at the plot boundary.
    replace_function(f,'landscapes',[
      ("ax.plot(par[j],par[i],'r*',ms=5)", "ax.plot(par[j],par[i],'r*',ms=5,clip_on=False,zorder=5)")])
    import performance_phase_figure_20260916 as pf
    replace_function(pf,'build',[
      ("bottom=.10,top=.94,hspace=.65,wspace=.64", "bottom=.08,top=.94,hspace=.70,wspace=.68"),
      ("fig.text(.55,.987,'Model fit, prediction and the limits of latent phase interpretation',ha='center',va='top',fontsize=10)", ""),
      ("'Temporal variance captured'", "'Temporal variance captured'"),
      ("'Small fit change'", "'Small score change'"),
      ("'Saved modulation peak (h)'", "'Saved peak (h)'"),
      ("'Refined modulation peak (h)'", "'Refined peak (h)'"),
      ("'Predicted preference phase (h)'", "'Predicted phase (h)'"),
      ("'Observed preference phase (h)'", "'Observed preference phase (h)'"),
      ("'Error reduction vs Random (%)'", "'Error reduction vs Random (%)'")])

def render_supplement_phases(bins):
    import model_phase_analysis as mp
    mp.save_table=no_table_write
    replace_function(mp,'make_figures',[
      ("gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.2],hspace=.60,wspace=.45)", "gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.2],left=.105,right=.95,top=.88,bottom=.12,hspace=.85,wspace=.80)"),
      ("gridspec_kw={'wspace':.46,'hspace':.85}", "gridspec_kw={'wspace':.70,'hspace':.95,'left':.105,'right':.88,'top':.86,'bottom':.08}"),
      ("fontsize=14,y=.995", "fontsize=10,y=.985"),
      ("fontsize=14,y=.99", "fontsize=10,y=.985"),
      ("label=f'{LABEL[a]}/{LABEL[b]} ({len(d)})'", "label=f'{LABEL[a]} / {\"NS\" if b.endswith(\"_no_swap\") else LABEL[b]} ({len(d)})'"),
      ("ax.legend(fontsize=7,loc='lower right')", "ax.legend(fontsize=8,loc='lower right',frameon=False,handlelength=.7,handletextpad=.25,labelspacing=.15)"),
      ("'A  Nominal reward-modulation peak'", "'A  Reward peak'"),
      ("'B  Maximum choice sensitivity β'", "'B  Peak sensitivity'"),
      ("'C  Reward parameter vs intake'", "'C  Reward vs intake'"),
      ("'D  Reward parameter vs preference'", "'D  Reward vs preference'"),
      ("'E  Predicted vs observed preference'", "'E  Predicted vs observed'"),
      ("'F  Phase changes across model structures'", "'F  Model dependence'"),
      ("f'G  Nuisance-optimized phase profiles (n={len(g)})'", "f'G  Phase profiles (n={len(g)})'"),
      ("'H  Phase-focused refinement in the same family'", "'H  Phase refinement'"),
      ("'Observed intake cosinor phase (h)'", "'Observed intake phase (h)'"),
      ("'Observed preference cosinor phase (h)'", "'Observed preference phase (h)'"),
      ("'Predicted preference cosinor phase (h)'", "'Predicted phase (h)'"),
      ("'Absolute reward-phase difference (h)'", "'Reward-phase difference (h)'"),
      ("'Cumulative fraction of animals'", "'Cumulative fraction'"),
      ("'Fixed modulation phase center (h)'", "'Fixed phase center (h)'"),
      ("'Animal order (saved phase)'", "'Animal order'"),
      ("'Saved modulation phase center (h)'", "'Saved peak (h)'"),
      ("'Phase after additional optimization (h)'", "'Refined peak (h)'"),
      ("f'n={len(g)}\\nmedian |Δphase|={np.median(delta):.2f} h'", "f'n={len(g)}; median\\n|Δphase|={np.median(delta):.2f} h'"),
      ("label='Small fit change'", "label='Δ score ≤2'"),
      ("label='Score improves >2'", "label='Δ score >2'"),
      ("'How to read the phases'", "'Model phase convention'"),
      ("ax.text(0,.88,'Reward peak center:\\n(6 − 12φ/π) mod 24 h\\n\\nβ maximum: +12 h\\n(imposed by the model).\\n\\nLatent, not measured.\\nClipping can flatten peaks.\\n\\nPA omitted.\\nDD: prior reference time.'", "ax.text(0,.82,'Reward: (6 − 12φ/π) mod 24 h\\nβ maximum: reward +12 h\\nLatent model quantities.\\nPA omitted; DD prior reference.'"),
      ("x=.03+(i%2)*.5;y=.13-(i//2)*.09", "x=.03+(i%2)*.55;y=.25-(i//2)*.16"),
      ("f'{chr(65+row*3)}  {date} {mouse} | {r.Condition}'", "f'{chr(65+row*3)}  {date}\\n{mouse} | {r.Condition}'"),
      ("f'{chr(66+row*3)}  Observed and predicted allocation\\n{LABEL[r.model]}'", "f'{chr(66+row*3)}  {LABEL[r.model]}\\nPreference'"),
      ("f'{chr(67+row*3)}  Latent modulation functions'", "f'{chr(67+row*3)}  Latent modulation'"),
      ("'Model reward term'", "'Reward term'"),
      ("'Choice sensitivity β'", "'Sensitivity β'")])
    mp.make_figures(bins[bins.modeling_cohort],f.read('model_phase_parameters'),f.read('model_phase_selected_animals'),f.read('model_phase_nuisance_profiles'),f.read('model_phase_profile_support'))

def render_supplement_light(bins):
    # Run only the manuscript figure section, never the diagnostic-report writer.
    import light_dark_audit as ld
    source=inspect.getsource(ld.make_figure)
    source=source[:source.index('    flagged=primary')]
    exec(compile(source,__file__+'::light_dark_figure','exec'),ld.__dict__)
    primary=f.read('light_dark_mouse_audit');primary=primary[primary.modeling_cohort]
    ld.make_figure(primary,bins,f.read('light_dark_sensitivity_model_counts'),f.read('light_dark_sensitivity_cv'))

def verify():
    from xml.etree import ElementTree as ET
    qa=DEST/'audit/figure_pdf_review';qa.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted((DEST/'figures').glob('Figure*.pdf')):
        doc=pdfium.PdfDocument(str(path));page=doc[0]
        width,height=page.get_width()*25.4/72,page.get_height()*25.4/72
        assert abs(width-175)<.001
        assert height<=210.001
        svg=ET.parse(path.with_suffix('.svg')).getroot();sizes=[];fonts=[]
        for t in svg.iter('{http://www.w3.org/2000/svg}text'):
            matches=[float(x) for c in t.iter() for x in re.findall(r'font-size:\s*([\d.]+)px',c.attrib.get('style',''))]
            if matches:sizes.append(max(matches))
        if sizes:assert min(sizes)>=8,(path.name,min(sizes))
        else:assert path.stem=='FigureS3',path.name
        tp=page.get_textpage();outside=[]
        for i in range(tp.count_chars()):
            if tp.get_text_range(i,1).isspace():continue
            left,bottom,right,top=tp.get_charbox(i)
            if left<-.5 or bottom<-.5 or right>page.get_width()+.5 or top>page.get_height()+.5:outside.append(i)
        assert not outside,(path.name,'Text extends outside page',outside)
        tp.close()
        bitmap=page.render(scale=2);im=bitmap.to_pil();im.save(qa/(path.stem+'.png'));im.close();bitmap.close();page.close();doc.close()
        row={'figure':path.stem,'width_mm':width,'height_mm':height,'minimum_body_font_pt':min(sizes) if sizes else None,'font':'Arial' if sizes else 'Original submitted raster artwork','out_of_page_characters':0,'files':{ext:digest(path.with_suffix('.'+ext)) for ext in ['pdf','png','svg']}}
        if path.stem in ['Figure1','Figure2','Figure3','Figure4']:
            tif=path.parent/(path.stem.replace('Figure','Fig')+'.tif')
            with Image.open(tif) as im:row['tiff']={'mode':im.mode,'dpi':[float(x) for x in im.info['dpi']],'pixels':im.size,'bytes':tif.stat().st_size};assert im.mode=='RGB' and im.info['dpi']==(600,600) and tif.stat().st_size<10_000_000
        rows.append(row)
    assert len(rows)==14
    (qa/'format_verification.json').write_text(json.dumps(rows,indent=2))
    return rows

def write_gallery():
    titles=[
      ('Figure1','Palatable-fluid preference and circadian rhythmicity'),
      ('Figure2','Intake, preference and phase organization'),
      ('Figure3','Simulated dynamics and empirical model comparison'),
      ('Figure4','Model fit and timing'),
      ('FigureS1','Genotype and fluid control trajectories'),
      ('FigureS2','Bottle controls and individual rhythmicity'),
      ('FigureS3','Observation process and model assumptions'),
      ('FigureS4','Simple and Bias conditional log-likelihood slices'),
      ('FigureS5','Reward conditional log-likelihood slices'),
      ('FigureS6','Full conditional log-likelihood slices'),
      ('FigureS7_reviewer_checks','Reviewer-requested validation checks'),
      ('FigureS8_trial_sensitivity','Assumed-event and score-scale sensitivity'),
      ('FigureS9_model_phases','Model phases and interpretation checks'),
      ('FigureS10_model_phase_examples','Measured behavior and model cycles')]
    body=['# Revised figure gallery',
      'All figures are **175 mm wide**. Newly generated plots use **Arial body text at least 8 pt**; S3 preserves the submitted artwork. All figures fit within a 210 mm height; exact dimensions are recorded in the format audit. PDF and SVG preserve editable/vector content; PNG previews are 300 dpi. The four main TIFFs are **600-dpi RGB/LZW**, each below 10 MB.',
      'The verified September 16 results are unchanged. Figure 4 presents the replacement performance and phase analyses; the former light-dark audit and earlier subgroup comparison are omitted. S3 preserves the original submitted artwork; old S10–S11 are now S9–S10. Full explanations and statistical details are in the manuscript captions.']
    for name,title in titles:
        short=name.split('_')[0].replace('FigureS','Supplement S').replace('Figure','Figure ')
        links=[f'[PDF]({name}.pdf)',f'[SVG]({name}.svg)',f'[PNG]({name}.png)']
        if name in ['Figure1','Figure2','Figure3','Figure4']:links.append(f'[TIFF for submission]({name.replace("Figure","Fig")}.tif)')
        body.extend([f'## {short}. {title}',' · '.join(links),f'![{short}: {title}]({name}.png)'])
    target=DEST/'figures/README.md';target.write_text('\n\n'.join(body)+'\n',encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',target.read_text(encoding='utf-8')):
        assert (target.parent/link).is_file(),link

def main():
    before=baseline_hashes()
    for name in ['figures','audit','reports']:(DEST/name).mkdir(parents=True,exist_ok=True)
    install_style()
    install_compact_style()
    from model_figure_polish import install as model_polish
    from supplement_figure_polish import install as supplement_polish
    from publication_polish import install as shared_polish
    shared_polish(f, globals())
    model_polish(f, globals())
    supplement_polish(f, globals())
    bins=f.pd.read_csv(BASE/'data/analysis_bins.csv')
    f.figure1(bins);f.figure2(bins);f.figure3();f.figure4(bins)
    f.supplement1(bins);f.supplement2(bins);f.supplement3();f.landscapes(bins);f.reviewer_supplement();f.trial_sensitivity_figure()
    render_supplement_phases(bins)
    rows=verify()
    write_gallery()
    after=baseline_hashes();assert before==after,'Frozen baseline changed'
    (DEST/'audit/figure_source_preservation.json').write_text(json.dumps({'baseline':str(BASE),'files':before,'all_preserved':True},indent=2))
    (DEST/'audit/figure_footer_legend_ledger.json').write_text(json.dumps(FOOTNOTES,indent=2))
    assert len(POLAR_QA)==8 and all(x['outline_radius']==x['individual_phase_radius']==1.0 and x['all_unclipped'] for x in POLAR_QA)
    (DEST/'audit/figure1_polar_boundary_verification.json').write_text(json.dumps(POLAR_QA,indent=2))
    modules={name:str(Path(module.__file__).resolve()) for name,module in list(sys.modules.items()) if getattr(module,'__file__',None) and Path(module.__file__).resolve().parent.name in ['scripts','code']}
    (DEST/'audit/figure_dependencies.json').write_text(json.dumps({'source_inputs':sorted(INPUTS),'source_modules':modules,'renderer':str(Path(__file__).resolve()),'omitted':['former FigureS9_light_dark_audit','FigureS12_exploratory_classification']},indent=2))
    print(f'Validated {len(rows)} figures; all {len(before)} baseline data/result/figure/sharing files unchanged.',flush=True)

if __name__=='__main__':main()
