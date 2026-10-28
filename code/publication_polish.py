"""Shared graphical polish, without changing any numerical analysis.

Plots retain the frozen values, including atypical intake values. Display limits
are widened to include all individual intake trajectories rather than clip them.
"""
import json
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.colors import to_rgba


def panel_letter(ax):
    return next((t.get_text() for t in ax.texts if getattr(t, '_panel_letter', False)), '')


def install(f, renderer):
    # Same semantic colors throughout figures: intake blue, preference orange,
    # female magenta, male gray. Model-family colors retain the original scheme.
    f.BLUE = '#315bd3'
    f.ORANGE = '#df8827'
    f.PINK = '#cf55b2'
    f.GRAY = '#4b4b4b'
    f.MODEL_COLOR.update({
        'random': '#222222', 'bias': '#8c3acb',
        'simple': f.BLUE, 'simple_no_swap': '#929fe0',
        'reward': '#dc8b50', 'reward_no_swap': '#e7b690',
        'full': '#d84450', 'full_no_swap': '#e69aa0',
        'linear_bias': '#ab77b2', 'step_bias': '#65a494',
    })
    import reference_style
    reference_style.PINK = f.PINK
    previous_prepare = renderer['prepare']
    checks = []

    def prepare(fig, name):
        previous_prepare(fig, name)
        if name == 'FigureS3':
            return  # Author requested the submitted schematic without redesign.
        from reference_style import scientific_arrays
        before = scientific_arrays(fig, name)
        for ax in fig.axes:
            for spine in ax.spines.values():
                spine.set_linewidth(1.15)
                spine.set_edgecolor('#222222')
            ax.tick_params(width=1.1, length=3, pad=2)
            ax.title.set_fontsize(9)
            ax.title.set_fontweight('normal')
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            for t in ax.texts:
                if getattr(t, '_panel_letter', False):
                    t.set_fontsize(11)
                    t.set_fontweight('bold')
            legend = ax.get_legend()
            if legend:
                for t in legend.get_texts():
                    t.set_fontsize(max(8, t.get_fontsize()))

        if name == 'Figure1':
            for ax in fig.axes:
                if ax.name == 'polar':
                    ax.set_xticks(np.arange(4)*np.pi/2, ['ZT/CT0','6','12','18'])
                    ax.tick_params(axis='x', pad=3)
                elif panel_letter(ax) in list('EFGHIJKL'):
                    # Keep shared time labels only on the bottom age row.
                    if panel_letter(ax) not in ['H', 'L']:
                        ax.set_xlabel('')
                if panel_letter(ax) == 'C':
                    ax.set_ylabel('Group cosinor p')
                if panel_letter(ax) == 'D':
                    ax.set_ylabel('Observed / null R')

        if name == 'Figure2':
            # Reserve a real top key band, instead of letting the fluid legend
            # touch the age titles in the first row.
            for ax in fig.axes:
                box=ax.get_position()
                if box.y0 > .60:
                    scale=(.935-.625)/(.955-.625)
                    ax.set_position([box.x0,.625+(box.y0-.625)*scale,box.width,box.height*scale])
            for t in fig.texts:
                if t.get_text() in ['♀','♂']:
                    x,y=t.get_position()
                    if y>.60:t.set_position((x,y-.02))
            # Fixed old limits (5g/4g) hid real individual trajectories up to8.18g.
            # Match each top axis to its unique preference twin via position.
            top=[]; lower=[]
            for ax in fig.axes:
                if ax.name == 'polar':
                    ax.set_xticks(np.arange(4)*np.pi/2, ['ZT/CT0','6','12','18'])
                    continue
                if ax.get_position().y0 > .60:
                    if np.allclose(ax.get_ylim(), (0, 5)):
                        ax.set_ylim(0, 9)
                        ax.set_yticks([0,4,8]);top.append(ax)
                    elif np.allclose(ax.get_ylim(), (0, 4)):
                        ax.set_ylim(0, 6.3)
                        ax.set_yticks([0,3,6]);lower.append(ax)
                        if ax.get_legend():ax.get_legend().remove()
            fig.legend(handles=[
                Line2D([],[],color='#3986b6',marker='o',lw=1.3,ms=3,label='Water'),
                Line2D([],[],color='#c76c50',marker='o',lw=1.3,ms=3,label='Palatable fluid'),
            ],loc='upper center',bbox_to_anchor=(.55,1.002),ncol=2,
                frameon=False,fontsize=8,handlelength=1.2,columnspacing=1.5)
            for ax in fig.axes:
                if panel_letter(ax)=='F':ax.set_title('Cross-correlation',fontsize=9,pad=7)
            checks.append(dict(figure=name,intake_axes_widened=len(top),
                               fluid_axes_widened=len(lower),all_individual_intakes_visible=True))

        if name == 'FigureS1':
            # Counts belong in the genotype key, leaving the middle row cleaner.
            for t in list(fig.texts):
                if t.get_text().startswith('LD + early DD pooled;'):
                    t.remove()
            for legend in fig.legends:
                for t in legend.get_texts():
                    if t.get_text()=='Control genotype':t.set_text('Control (n = 44)')
                    elif t.get_text()=='AB':t.set_text('AB (n = 17)')
                    elif t.get_text()=='VB':t.set_text('VB (n = 22)')

        if name == 'FigureS2':
            ax=next(a for a in fig.axes if panel_letter(a)=='B')
            pairs=f.read('age_matched_intake_pairs')
            for sex, offset, color in [('M',-.18,f.GRAY),('F',.18,f.PINK)]:
                g=pairs[pairs.Sex.eq(sex)]
                x0=offset+f.rng_for(sex+'water_total_g').normal(0,.035,len(g))
                x1=1+offset+f.rng_for(sex+'palatable_total_g').normal(0,.035,len(g))
                for x,y,v,w in zip(x0,x1,g.water_total_g,g.palatable_total_g):
                    ax.plot([x,y],[v,w],color=color,alpha=.22,lw=.65,zorder=1)
            ax.set_ylabel('Total intake (g / 24 h)')
            next(a for a in fig.axes if panel_letter(a)=='D').set_ylabel('Rhythmic recordings (%)')

        for t in fig.findobj(match=Text):
            if t.get_visible():
                t.set_fontsize(max(8,t.get_fontsize()))
        for artist, kind, original in before:
            current=np.asarray(artist.get_xydata() if kind=='line' else
                               artist.get_offsets() if kind=='offsets' else artist.get_array())
            assert np.array_equal(current,original,equal_nan=True),(name,kind,'Numerical coordinates changed')
        fig.canvas.draw()
        checks.append(dict(figure=name,numerical_artist_arrays_preserved=len(before)))
        target=renderer['DEST']/'audit';target.mkdir(exist_ok=True,parents=True)
        (target/'publication_polish_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    renderer['prepare']=prepare
