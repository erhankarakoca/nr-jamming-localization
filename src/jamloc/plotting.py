"""Shared figure style: IEEE column width, colorblind-safe colors plus markers per attack.

Figures are built with ``matplotlib.figure.Figure`` directly (no pyplot), so they render the
same way on every OS and on headless machines such as Colab.
"""

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

COLUMN_WIDTH = 3.5  # IEEE two-column layout, single column [inch]

# Okabe-Ito colors; every attack also has its own marker and line style
ATTACK_STYLE = {
    "none": {"color": "#000000", "marker": "x", "linestyle": ":"},
    "pilot": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "data": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    "partial": {"color": "#009E73", "marker": "^", "linestyle": "-."},
}

RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,  # embed TrueType fonts (editable text in the PDF)
    "savefig.dpi": 300,
}


def paper_style():
    """Context manager: everything drawn inside it uses the paper style."""
    return mpl.rc_context(RC)


def new_figure(width: float = COLUMN_WIDTH, height: float = 2.4, **subplot_kw):
    """Figure and axes; call inside ``paper_style()`` so labels get the paper fonts."""
    fig = Figure(figsize=(width, height), layout="constrained")
    return fig, fig.subplots(**subplot_kw)


def save(fig: Figure, path_without_suffix: Path) -> None:
    """Save as PDF (for the paper) and PNG (for slides and quick viewing)."""
    with paper_style():
        for suffix in (".pdf", ".png"):
            fig.savefig(path_without_suffix.with_suffix(suffix))
