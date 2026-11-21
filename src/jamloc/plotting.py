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
    # Times-like text and math, as in the IEEEtran body text. STIX ships with matplotlib, so the
    # figures look the same on every machine.
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "legend.fancybox": False,
    "legend.edgecolor": "0.3",
    "legend.framealpha": 1.0,
    "legend.borderpad": 0.3,
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


def plot_bler(ax, x, errors, blocks, style: dict, label: str | None = None) -> None:
    """BLER curve with its 95% confidence band; points without any block error are shown as a
    hollow downward triangle at their 95% upper bound instead of disappearing from the log axis.
    """
    import numpy as np

    from jamloc.metrics import wilson_interval

    x, errors, blocks = (np.asarray(v, float) for v in (x, errors, blocks))
    lo, hi = wilson_interval(errors, blocks)
    seen = errors > 0
    color = style.get("color", "black")
    ax.plot(x[seen], errors[seen] / blocks[seen], label=label, **style)
    ax.fill_between(x[seen], np.maximum(lo[seen], 1e-6), hi[seen], color=color, alpha=0.15, lw=0)
    ax.plot(x[~seen], hi[~seen], ls="none", marker="v", mfc="none", mec=color, ms=4)


def zero_error_handle():
    """Legend entry explaining the hollow triangles drawn by ``plot_bler``."""
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        ls="none",
        marker="v",
        mfc="none",
        mec="black",
        ms=4,
        label="no block error: 95% upper bound",
    )
