"""Physical publication sizing shared by every manuscript figure.

The page, rather than its subsequently cropped bounding box, is 175 mm wide.
Text sizes are points at that final physical size; no export scaling is applied.
"""
import textwrap
import numpy as np
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

WIDTH_MM = 175.0
WIDTH_IN = WIDTH_MM / 25.4
DPI = 300
FONT = 'DejaVu Sans'

# Preserve sufficient vertical space as horizontal grids become journal width.
HEIGHTS = {'Figure1': 9.1, 'Figure2': 12.4, 'Figure3': 11.3,
           'Figure4': 10.3, 'FigureS1': 7.2, 'FigureS2': 8.2,
           'FigureS3': 9.0, 'FigureS4': 7.7, 'FigureS5': 8.5,
           'FigureS6': 9.1, 'FigureS7_reviewer_checks': 11.5,
           'FigureS8_trial_sensitivity': 8.1,
           'FigureS9_light_dark_audit': 10.2,
           'FigureS10_model_phases': 12.5,
           'FigureS11_model_phase_examples': 12.2}

def _wrap(text, width):
    """Wrap prose only; keep math and existing deliberate line breaks intact."""
    if '$' in text:
        return text
    return '\n'.join(textwrap.fill(line, width=max(10, width),
                                   break_long_words=False, break_on_hyphens=False)
                     for line in text.split('\n'))

def prepare(fig, name):
    if getattr(fig, '_publication_prepared', False):
        return
    fig.set_size_inches(WIDTH_IN, HEIGHTS.get(name, max(7., fig.get_figheight())), forward=True)
    fig.set_dpi(DPI)
    # All readable content, including legends and tick labels, is at least 8 pt.
    for text in fig.findobj(match=Text):
        text.set_fontfamily(FONT)
        text.set_fontsize(8)
    for ax in fig.axes:
        width_pt = ax.get_position().width * WIDTH_IN * 72
        title = ax.get_title(loc='left') or ax.get_title()
        if ax.get_title(loc='left'):
            ax.set_title('', loc='left')
        if title:
            ax.set_title(_wrap(title, int(width_pt / 4.5)), loc='center', fontsize=9, pad=8)
        ax.xaxis.label.set_text(_wrap(ax.xaxis.label.get_text(), int(width_pt / 4.1)))
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
        for t in ax.texts:
            if getattr(t, '_panel_letter', False):
                t.set_fontsize(11)
            elif t.get_transform() == ax.transAxes and len(t.get_text()) > 34:
                t.set_text(_wrap(t.get_text(), int(width_pt / 4.0)))
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(8)
                text.set_text(_wrap(text.get_text(), max(13, int(width_pt / 4.4))))
        if ax.name != 'polar' and name in {'FigureS4','FigureS5','FigureS6'} and not hasattr(ax,'_colorbar'):
            ax.xaxis.set_major_locator(MaxNLocator(2))
            ax.yaxis.set_major_locator(MaxNLocator(2))
    for text in fig.texts:
        if text is getattr(fig, '_suptitle', None):
            text.set_fontsize(10)
            text.set_text(_wrap(text.get_text(), 77))
        else:
            text.set_text(_wrap(text.get_text(), 103))
            # Footnotes are inside the page and grow upward from this baseline.
            if text.get_position()[1] < .04:
                text.set_va('bottom')
    fig._publication_prepared = True
