"""Final presentation of main model figures; preserves every plotted data array.

Installed after the historical renderer's layout modules. It changes only axes
positions, labels, graphical keys and figure height. Saved fits are not modified.
"""
import json
import numpy as np
from matplotlib.lines import Line2D


def axis_by_letter(fig, letter):
    matches = [ax for ax in fig.axes if any(
        getattr(t, '_panel_letter', False) and t.get_text() == letter
        for t in ax.texts)]
    if len(matches) != 1:
        raise ValueError(f'Expected one panel {letter}, found {len(matches)}')
    return matches[0]


def letter_position(ax, left, top):
    for text in ax.texts:
        if getattr(text, '_panel_letter', False):
            text.set_transform(ax.figure.transFigure)
            text.set_position((left, top))
            text.set_va('bottom')
            text.set_fontsize(11)
            text.set_fontweight('bold')


def model_figures(fig, name, f):
    if name == 'Figure3':
        # The 50/500-run distinction is explained by the caption, not a subtitle.
        axis_by_letter(fig, 'B').set_title('Circadian profiles', fontsize=9, pad=7)
        for letter in ['C', 'D', 'E', 'F', 'G']:
            ax = axis_by_letter(fig, letter)
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_xlabel('ZT (h)' if letter in ['D', 'G'] else '')
        # One consistently short name across figures and keys.
        axis_by_letter(fig, 'F').set_title('Circadian reward RL', fontsize=9, pad=6)
        axis_by_letter(fig, 'J').set_title('Individual model score differences', fontsize=9, pad=6)
        return

    if name != 'Figure4':
        return

    boxes = {
        'A': [.16, .755, .32, .185],
        'B': [.65, .755, .32, .185],
        'C': [.16, .455, .32, .185],
        'D': [.65, .455, .32, .185],
        'E': [.16, .145, .32, .195],
        'F': [.65, .145, .32, .195],
    }
    for letter, box in boxes.items():
        ax = axis_by_letter(fig, letter)
        ax.set_position(box)
        letter_position(ax, box[0] - .037, box[1] + box[3] + .015)
        ax.title.set_fontsize(9)
        ax.title.set_position((.5, 1.0))
    axis_by_letter(fig, 'C').set_title('Cross-validation', fontsize=9, pad=8)
    axis_by_letter(fig, 'A').set_xlabel('Error reduction vs Random (%)')
    axis_by_letter(fig, 'B').set_xlabel('Variation explained (%)')
    axis_by_letter(fig, 'C').set_xlabel('Prediction error (MSE)')

    # D's colors describe search improvement, unlike the model-color coding in E/F.
    d = axis_by_letter(fig, 'D')
    d.legend(
        handles=[
            Line2D([], [], marker='o', ms=4, linestyle='none', color='#b04b36'),
            Line2D([], [], marker='o', ms=4, linestyle='none', color='.6'),
        ],
        labels=['Improved search', 'Other'],
        loc='upper left', bbox_to_anchor=(.03, .84), borderaxespad=0,
        frameon=False, fontsize=8, handlelength=.8, handletextpad=.35,
        labelspacing=.2,
    )

    # These models were selected separately for each animal. Both panels use the
    # same selected model for a given animal; open/filled is EMPIRICAL significance.
    models = ['reward', 'reward_no_swap', 'full', 'full_no_swap']
    fig.legend(
        handles=[Line2D([], [], marker='o', markersize=4, linestyle='none',
                        color=f.MODEL_COLOR[m]) for m in models],
        labels=['Reward', 'Reward NS', 'Full', 'Full NS'],
        loc='lower center', bbox_to_anchor=(.55, .053), ncol=4,
        frameon=False, fontsize=8, handlelength=.8, handletextpad=.35,
        columnspacing=1.1, borderaxespad=0,
    )
    fig.legend(
        handles=[
            Line2D([], [], marker='o', markersize=4, linestyle='none',
                   color='.3', markerfacecolor='.3'),
            Line2D([], [], marker='o', markersize=4, linestyle='none',
                   color='.3', markerfacecolor='white'),
        ],
        labels=['Measured rhythm: p < 0.05', 'p ≥ 0.05'],
        loc='lower center', bbox_to_anchor=(.55, .018), ncol=2,
        frameon=False, fontsize=8, handlelength=.8, handletextpad=.35,
        columnspacing=1.0, borderaxespad=0,
    )


def install(f, renderer):
    renderer['HEIGHT_MM']['Figure4'] = 185
    previous_prepare = renderer['prepare']
    checks = []

    def prepare(fig, name):
        previous_prepare(fig, name)
        if name not in ['Figure3', 'Figure4']:
            return
        from reference_style import scientific_arrays
        before = scientific_arrays(fig, name)
        model_figures(fig, name, f)
        for artist, kind, original in before:
            current = np.asarray(artist.get_xydata() if kind == 'line' else
                                 artist.get_offsets() if kind == 'offsets' else
                                 artist.get_array())
            if not np.array_equal(current, original, equal_nan=True):
                raise AssertionError((name, kind, 'Data changed during polish'))
        fig.canvas.draw()
        checks.append(dict(figure=name, data_arrays_preserved=len(before),
                           width_mm=fig.get_figwidth()*25.4,
                           height_mm=fig.get_figheight()*25.4,
                           model_color_key=name == 'Figure4',
                           measured_rhythm_fill_key=name == 'Figure4'))
        target = renderer['DEST'] / 'audit'
        target.mkdir(exist_ok=True, parents=True)
        (target/'model_figure_polish_verification.json').write_text(
            json.dumps(checks, indent=2), encoding='utf-8')

    renderer['prepare'] = prepare
