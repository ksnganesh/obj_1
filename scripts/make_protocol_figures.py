"""Schematics of the DI-QDS protocol (Figs. 1 and 2 of the article), drawn from scratch with Matplotlib.

    fig_protocol_distribution.png   (a) three parties, two independent device-independent links, where trusted
                                     randomness enters; (b) the seven steps of each key-generation protocol
                                     (distribution stage, Fan et al. Sec. II A; src/diqds/protocol/distribution.py)
    fig_protocol_messaging.png      the messaging stage (Sec. II B; src/diqds/protocol/messaging.py): who sends what
                                     to whom, in order, with the forgery and repudiation conditions

    python scripts/make_protocol_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diqds.plotting import INK, INK_2, SERIES, SURFACE, setup  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RES = ROOT / "results"
ALICE, BOB, CHARLIE, RAND = SERIES[0], SERIES[1], SERIES[2], SERIES[3]
LINE = "#8c8a85"


def tint(hex_color: str, a: float) -> tuple:
    """Opaque mix of a colour with the surface (a = share of the colour)."""
    c = list(hex_color) if not isinstance(hex_color, str) else [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    s = [int(SURFACE[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(a * ci + (1 - a) * si for ci, si in zip(c, s))


def box(ax, x, y, w, h, text="", fc=SURFACE, ec=INK_2, lw=0.9, fs=8.0, color=INK, weight="normal", r=1.2,
        ha="center", va="center", tx=None, ty=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                linestyle=ls))
    if text:
        ax.text(x + w / 2 if tx is None else tx, y + h / 2 if ty is None else ty, text, ha=ha, va=va, fontsize=fs,
                color=color, fontweight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK_2, lw=1.1, style="-|>", ls="-", ms=9, shrink=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, shrinkA=shrink, shrinkB=shrink))


def tag(ax, x, y, text, fc, fs=6.9, w=None):
    w = w if w is not None else 0.58 * len(text) + 1.6
    box(ax, x - w / 2, y - 1.15, w, 2.3, text, fc=tint(fc, 0.22), ec=tint(fc, 0.9), lw=0.7, fs=fs, r=0.9)


# ------------------------------------------------------------------------------------------------------- Fig. 1
def fig_distribution() -> None:
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")

    # ---------------------------------------------------------------- (a) architecture
    ax.text(1.2, 71.3, "(a)  Three parties, two independent device-independent links", fontsize=9.2,
            fontweight="bold", va="top", color=INK)
    labs = {"Bob": (2.0, BOB), "Alice": (38.5, ALICE), "Charlie": (77.0, CHARLIE)}
    widths = {"Bob": 21.0, "Alice": 23.0, "Charlie": 21.0}
    y0, h = 38.5, 22.0
    for name, (x, c) in labs.items():
        w = widths[name]
        box(ax, x, y0, w, h, fc=tint(c, 0.07), ec=c, lw=1.1, r=1.6)
        ax.text(x + 1.3, y0 + h - 1.4, name, fontsize=9.0, fontweight="bold", color=INK, va="top")
    # devices
    for xd, lab in ((13.0, "$D_B$"), (40.0, "$D_A^{(B)}$"), (51.5, "$D_A^{(C)}$"), (78.5, "$D_C$")):
        box(ax, xd, 47.5, 8.5, 6.5, "device\n" + lab, fc=SURFACE, ec=INK_2, fs=7.6)
    ax.text(50.0, 45.6, "separate devices per link", fontsize=7.0, color=INK_2, ha="center", va="top")
    # heralding stations; photons from both devices interfere there
    for xc, xl, xr in ((29.25, 21.5, 40.0), (69.75, 60.0, 78.5)):
        box(ax, xc - 3.6, 48.6, 7.2, 4.3, "herald\nstation", fc=tint(INK_2, 0.08), ec=INK_2, fs=6.7)
        arrow(ax, xl, 50.75, xc - 3.6, 50.75, color=LINE, lw=0.9, ls=(0, (3, 2)), ms=7)
        arrow(ax, xr, 50.75, xc + 3.6, 50.75, color=LINE, lw=0.9, ls=(0, (3, 2)), ms=7)
        ax.text(xc, 47.9, "heralds\nentanglement", fontsize=6.6, color=INK_2, ha="center", va="top",
                linespacing=1.2)
    # trusted randomness: private random-number generators
    tag(ax, 8.4, 41.6, "RNG: $Y,\\,T$", RAND, fs=7.1, w=10.0)
    tag(ax, 50.0, 41.4, "RNG: $X,\\,T,\\,F$", RAND, fs=7.1, w=12.0)
    tag(ax, 91.6, 41.6, "RNG: $Y,\\,T$", RAND, fs=7.1, w=10.0)
    # classical channels
    ax.plot([12.5, 12.5, 87.5, 87.5], [60.5, 64.6, 64.6, 60.5], color=LINE, lw=1.0, zorder=1)
    ax.plot([50.0, 50.0], [60.5, 64.6], color=LINE, lw=1.0, zorder=1)
    ax.text(31.25, 64.6, "authenticated classical channels", fontsize=7.0, color=INK_2, ha="center", va="center",
            bbox=dict(fc=SURFACE, ec="none", pad=1.2), zorder=2)
    ax.text(68.75, 64.6, "public seeds for hashing", fontsize=7.0, color=INK_2, ha="center", va="center",
            bbox=dict(fc=SURFACE, ec="none", pad=1.2), zorder=2)
    ax.text(29.25, 57.3, "link A–B  →  $K_{AB}$", fontsize=7.8, color=INK, ha="center")
    ax.text(69.75, 57.3, "link A–C  →  $K_{AC}$", fontsize=7.8, color=INK, ha="center")

    # ---------------------------------------------------------------- (b) key-generation protocol
    ax.text(1.2, 34.6, "(b)  Distribution stage: the key-generation protocol run on each link", fontsize=9.2,
            fontweight="bold", va="top", color=INK)
    steps = [("1", "Parameters", "$n,\\ \\gamma,\\ q$\n$EC_{\\max},\\ \\ell_{key}$"),
             ("2", "Data\nacquisition", "$n$ heralded rounds\n$X, Y, T \\rightarrow A, B$"),
             ("3", "Sifting", "key rounds:\n$X = Y$ (2 settings)"),
             ("4", "Noisy pre-\nprocessing", "flip each key\nbit w.p. $q$"),
             ("5", "Error\ncorrection", "Cascade, $\\leq EC_{\\max}$\n64-bit tag"),
             ("6", "Parameter\nestimation", "CHSH wins and\nlosses accepted?"),
             ("7", "Privacy\namplification", "Toeplitz hash\nto $\\ell_{key}$ bits")]
    x, w, gap, yb, hb = 1.0, 12.3, 1.45, 15.0, 14.0
    centers = []
    for i, (num, title, body) in enumerate(steps):
        box(ax, x, yb, w, hb, fc=tint(ALICE, 0.06), ec=INK_2, lw=0.9, r=1.3)
        ax.text(x + 0.9, yb + hb - 0.9, num, fontsize=8.8, fontweight="bold", color=ALICE, va="top")
        ax.text(x + w / 2 + 0.7, yb + hb - 1.0, title, fontsize=7.6, fontweight="bold", color=INK, ha="center",
                va="top", linespacing=1.12)
        ax.text(x + w / 2, yb + 3.7, body, fontsize=6.9, color=INK, ha="center", va="center", linespacing=1.3)
        centers.append(x + w / 2)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.15, yb + hb / 2, x + w + gap - 0.15, yb + hb / 2, ms=8)
        x += w + gap
    ax.text(99.6, yb + hb / 2 + 0.2, "$K_{AB}$\n$K_{AC}$", fontsize=8.2, color=INK, ha="right", va="center",
            fontweight="bold")
    # what each step consumes (yellow: trusted randomness; grey: authenticated messages)
    tag(ax, centers[1], 11.6, "private $X, Y, T$", RAND, fs=7.0, w=11.6)
    tag(ax, centers[3], 11.6, "private flips $F$", RAND, fs=7.0, w=11.6)
    tag(ax, centers[4], 11.6, "public seed", RAND, fs=7.0, w=9.4)
    tag(ax, centers[6], 11.6, "public seed", RAND, fs=7.0, w=9.4)
    for c in (centers[2], centers[4], centers[5], centers[6]):
        tag(ax, c, 8.0, "authenticated", INK_2, fs=7.0, w=10.8)
    ax.text(centers[4], 3.6, "abort if more than\n$EC_{\\max}$ or tag differs", fontsize=6.8, color=INK_2,
            ha="center", va="center")
    ax.text(centers[5], 3.6, "abort if the counts\nfail the test", fontsize=6.8, color=INK_2, ha="center",
            va="center")
    ax.text(1.0, 3.4, "under each step: the trusted randomness it consumes (yellow)\nand its authenticated "
            "classical messages (grey)", fontsize=6.8, color=INK_2, va="center", linespacing=1.3)
    fig.savefig(RES / "fig_protocol_distribution.png", dpi=220)
    fig.savefig(RES / "fig_protocol_distribution.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- Fig. 2
def fig_messaging() -> None:
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    xs = {"Alice": 17.0, "Bob": 50.0, "Charlie": 83.0}
    cols = {"Alice": ALICE, "Bob": BOB, "Charlie": CHARLIE}
    keys = {"Alice": "$X_a = X_c,\\ Y_a = Y_b \\oplus Y_c,\\ Z_a = Z_c$\nfrom her copies of $K_{AB}$, $K_{AC}$",
            "Bob": "$(X_b,\\ Y_b,\\ Z_b)$ from $K_{AB}$",
            "Charlie": "$(X_c,\\ Y_c,\\ Z_c)$ from $K_{AC}$"}
    for name, x in xs.items():
        box(ax, x - 15.5, 88.0, 31.0, 11.2, fc=tint(cols[name], 0.10), ec=cols[name], lw=1.1, r=1.4)
        ax.text(x, 97.6, name, fontsize=9.2, fontweight="bold", ha="center", va="top", color=INK)
        ax.text(x, 92.3, keys[name], fontsize=7.2, ha="center", va="center", color=INK, linespacing=1.35)
        ax.plot([x, x], [16.0, 88.0], color=LINE, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax.text(50.0, 85.2, "one signature uses $3n_s = 138$ key bits of each link;  $f(Z)$: irreducible polynomial "
            "from $Z$;  $H$: LFSR-based Toeplitz hash", fontsize=7.0, color=INK_2, ha="center", va="center",
            bbox=dict(fc=SURFACE, ec="none", pad=1.5), zorder=3)

    def local(x, y, w, h, title, body, c):
        box(ax, x - w / 2, y - h / 2, w, h, fc=tint(c, 0.12), ec=c, lw=1.0, r=1.2)
        ax.text(x, y + h / 2 - 1.1, title, fontsize=7.9, fontweight="bold", ha="center", va="top", color=INK)
        ax.text(x, y - 1.5, body, fontsize=7.3, ha="center", va="center", color=INK, linespacing=1.35)

    def msg(x0, x1, y, text, sub):
        arrow(ax, x0, y, x1, y, color=INK_2, lw=1.2, ms=10)
        ax.text((x0 + x1) / 2, y + 1.0, text, fontsize=7.6, ha="center", va="bottom", color=INK)
        ax.text((x0 + x1) / 2, y - 1.0, sub, fontsize=6.7, ha="center", va="top", color=INK_2)

    local(xs["Alice"], 75.5, 30.0, 9.6, "1  Signing of Alice",
          "$Dig = H(f(Z_a), X_a)\\cdot Doc$\n$Sig = Dig \\oplus Y_a$", ALICE)
    msg(xs["Alice"] + 0.6, xs["Bob"] - 0.6, 65.5, "$\\{Sig,\\ Doc\\}$", "public channel")
    msg(xs["Bob"] + 0.6, xs["Charlie"] - 0.6, 58.5, "$\\{Sig,\\ Doc,\\ Y_b\\}$", "authenticated; Charlie records it")
    msg(xs["Charlie"] - 0.6, xs["Bob"] + 0.6, 51.5, "$\\{X_c,\\ Y_c,\\ Z_c\\}$", "sent only after recording")
    local(xs["Bob"], 41.5, 31.0, 11.0, "2  Verification of Bob",
          "expected $Sig \\oplus Y_b \\oplus Y_c$\n= actual $H(f(Z_c), X_c)\\cdot Doc$ ?", BOB)
    msg(xs["Bob"] + 0.6, xs["Charlie"] - 0.6, 31.5, "accept", "otherwise reject and abort")
    local(xs["Charlie"], 22.5, 31.0, 11.0, "3  Certification of Charlie",
          "same test with the $Y_b$\nBob forwarded: accept iff equal", CHARLIE)
    for y, s in ((75.5, "step 1"), (55.0, "step 2"), (22.5, "step 3")):
        ax.text(0.5, y, s, fontsize=6.8, color=INK_2, rotation=90, va="center", ha="left")
    # forgery point on Bob's forward
    ax.plot([54.6], [58.5], marker="o", ms=10.0, mfc=SURFACE, mec=BOB, mew=1.5, zorder=5)
    ax.text(54.6, 58.5, "F", fontsize=7.0, ha="center", va="center", color=INK, fontweight="bold", zorder=6)
    for x0, c, title, body in (
            (1.5, BOB, "F   Forgery (dishonest Bob)",
             "forwards $\\{Doc \\oplus \\Delta,\\ Sig \\oplus t\\}$ instead; Charlie accepts\n"
             "iff $t = H(f(Z_c), X_c)\\cdot \\Delta$, which needs $X_c$ and $Z_c$:\n"
             "they stay secret only through privacy amplification (Sec. VI)"),
            (51.5, ALICE, "Repudiation (dishonest Alice)",
             "Bob and Charlie test the same $\\{Sig, Doc, Y_b, X_c, Y_c, Z_c\\}$,\n"
             "so they can disagree only if a key-exchange message\nis corrupted: $\\varepsilon_{rep} = 2\\varepsilon$")):
        box(ax, x0, 0.8, 47.0, 13.6, fc=SURFACE, ec=tint(c, 0.9), lw=0.9, r=1.2)
        ax.text(x0 + 1.5, 13.4, title, fontsize=7.8, fontweight="bold", va="top", color=INK)
        ax.text(x0 + 1.5, 5.6, body, fontsize=7.0, va="center", color=INK, linespacing=1.4)
    fig.savefig(RES / "fig_protocol_messaging.png", dpi=220)
    fig.savefig(RES / "fig_protocol_messaging.pdf")
    plt.close(fig)


def main() -> None:
    setup()
    plt.rcParams.update({"axes.grid": False, "mathtext.default": "it"})
    RES.mkdir(exist_ok=True)
    fig_distribution()
    fig_messaging()
    print("written", RES / "fig_protocol_distribution.png", RES / "fig_protocol_messaging.png")


if __name__ == "__main__":
    main()
