"""Figures for notebook 03, Context Engineering.

This script draws one of the four pictures in this folder:

- `context_parts.png`: the six parts of a single call to the model, stacked in the
  order they are sent, each with a note on how often it changes.

`chain_two_steps.png`, `memory_strategies.png` and `rag_pipeline.png` came with the
repository and have no source here, so running this script regenerates one file, not
the folder.

The band for the examples sits between the knowledge and the conversation history,
because that is where example turns go in the message list: they are presented to the
model as earlier turns, ahead of the real ones. It is also the order the notebook
teaches the parts in, knowledge in section 2, examples in section 3, the history in
sections 4 and 5.

The bands are deliberately not all the same height. Knowledge and the conversation
history are drawn tallest because they are the two parts that grow, which is the point
section 2 measures and section 5 works around.

The palette is written out below rather than read from a theme file, because this
repository is published on its own and has no path to one.

Run from anywhere:

    uv run --no-project --with matplotlib python figures.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parent

INK = "#1A1A1A"  # the lettering inside the bands
MUTED = "#777777"  # the column heading, the arrow and the caption

plt.rcParams.update({
    # Pinned rather than left to the default, so the picture is the same on any machine
    # and keeps the lettering of the three figures that came with the repository.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

W, H = 19.0, 11.7  # inches; at 100 dpi this is 1900 x 1170 px
BOX_L, BOX_R = 1.72, 8.90  # the stack of bands
ARROW_X = 1.05  # the "sent in this order" arrow, to the left of the stack
CHANGES_X = 9.42  # the right-hand column
TOP = H - 1.28  # where the first band starts
GAP = 0.08  # between two bands

# title, subtitle, height, edge color, fill color, how often this part changes
BANDS = [
    ("system prompt", "role, rules, output format", 1.26,
     "#333333", "#FFFFFF", "rarely"),
    ("knowledge", "retrieved passages or a whole document", 1.93,
     "#E08214", "#FDF2E4", "every question"),
    ("examples", "an input and the answer you want", 1.30,
     "#7A52A1", "#F4EFF9", "rarely"),
    ("conversation history", "earlier questions and answers", 2.20,
     "#2171B5", "#E8F1FB", "every turn: grows"),
    ("tool results", "answers of a search or a database (week 7)", 1.24,
     "#555555", "#EFEFEF", "every call"),
    ("question", "what the user just wrote", 1.22,
     "#227A3D", "#EDF6EE", "every turn"),
]

CAPTION = ("Everything the model should use has to be in the context; "
           "every token in it is paid for.")


def context_parts():
    """Draw the stack of context parts and write it to context_parts.png."""
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # full bleed: no subplot margins
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(0.5 * (BOX_L + BOX_R), H - 0.63, "one call to the model",
            ha="center", va="center", fontsize=20, fontweight="bold", color=INK)
    ax.text(CHANGES_X, H - 0.63, "changes",
            ha="left", va="center", fontsize=20, fontweight="bold", color=MUTED)

    y = TOP
    for title, subtitle, height, edge, face, changes in BANDS:
        bottom = y - height
        ax.add_patch(FancyBboxPatch(
            (BOX_L, bottom), BOX_R - BOX_L, height,
            boxstyle="round,pad=0,rounding_size=0.18",
            facecolor=face, edgecolor=edge, linewidth=2.6,
        ))
        mid = bottom + height / 2
        ax.text(0.5 * (BOX_L + BOX_R), mid + 0.21, title,
                ha="center", va="center", fontsize=19, color=INK)
        ax.text(0.5 * (BOX_L + BOX_R), mid - 0.21, subtitle,
                ha="center", va="center", fontsize=18, color=INK)
        # The right-hand column takes the band's own color, except for the two drawn
        # in near-black and grey, where a colored note would look like a mistake.
        ax.text(CHANGES_X, mid, changes, ha="left", va="center", fontsize=18,
                color=INK if edge == "#333333" else edge)
        y = bottom - GAP

    stack_bottom = y + GAP
    ax.annotate("", xy=(ARROW_X, stack_bottom - 0.03), xytext=(ARROW_X, TOP),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.5",
                                color="#888888", linewidth=2.2))
    ax.text(ARROW_X - 0.42, 0.5 * (TOP + stack_bottom), "sent in this order",
            ha="center", va="center", fontsize=16, color="#888888", rotation=90)

    ax.text(9.50, 0.30, CAPTION, ha="center", va="center", fontsize=18, color=MUTED)

    fig.savefig(OUT / "context_parts.png", dpi=100, facecolor="white")
    plt.close(fig)
    print("context_parts.png")


if __name__ == "__main__":
    context_parts()
