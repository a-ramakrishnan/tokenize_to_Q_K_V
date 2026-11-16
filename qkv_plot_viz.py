#!/usr/bin/env python3
# Ramakrishnan A
# File 8-B in the series on Building LLMs from scratch
"""
qkv_plot_viz.py — watch Q, K and V change while a tiny attention head trains,
rendered as matplotlib heatmaps, one training step per keypress.

Two scopes, chosen at startup:

  1  SINGLE WORD    the loss comes from ONE word predicting the word after it.
                    W_Q/W_K/W_V are shaped by that single query row, and the
                    panels highlight it: its query vector, its attention over
                    the words it can see, its prediction.

  2  FULL SENTENCE  the loss comes from every position at once. Same panels,
                    but the whole attention matrix and all 8 predictions.

Everything except W_Q, W_K, W_V is frozen — the embeddings, the positional
vectors, W_O, the unembedding — so every number that moves on screen moved
because of those three matrices.

Under the panels, a "what the last step did" band reads the movement back in
plain language: which weight moved most and which column of Q/K/V it dragged
with it, which attention weight shifted and what happened to the raw score
behind it, and an exact split of the change in what a word reads out into
"looked somewhere else" versus "V itself moved".

For sentences under 5 words there is also a bottom block with the exact
arithmetic: the per-position loss terms, the gradient matrices dL/dW_Q,
dL/dW_K, dL/dW_V, the chain rule that produced them, and Adam's update
followed through on the single largest-gradient weight, checked against the
change actually applied.

The sentence, the scope and the query word are all asked for at startup;
any of them can be supplied as a flag to skip that prompt.

  python3 qkv_plot_viz.py
  python3 qkv_plot_viz.py --sentence "the dog chased the ball across the park"
  python3 qkv_plot_viz.py --scope word --word brown
"""

import argparse
import math
import sys
import textwrap

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

# --------------------------------------------------------------- fit to screen --
FS = 1.0            # global font scale, set once the figure size is known
FIG_SCALE = 1.0     # user nudge via --fig-scale
LEGIBLE_FS = 0.68   # below this the arithmetic block is too small to read


def fs(v):
    return v * FS


_SCREEN_PX = None
FALLBACK_SCREEN_PX = (1440, 900)


def _screen_px_darwin():
    """CoreGraphics through ctypes — no Tk, no PyObjC, no subprocess."""
    import ctypes
    import ctypes.util
    path = (ctypes.util.find_library("ApplicationServices")
            or "/System/Library/Frameworks/ApplicationServices.framework"
               "/ApplicationServices")
    cg = ctypes.cdll.LoadLibrary(path)
    cg.CGMainDisplayID.restype = ctypes.c_uint32
    cg.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]
    cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
    cg.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
    cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
    d = cg.CGMainDisplayID()
    return int(cg.CGDisplayPixelsWide(d)), int(cg.CGDisplayPixelsHigh(d))


def _screen_px_windows():
    import ctypes
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def _screen_px_x11():
    import re
    import subprocess
    out = subprocess.run(["xrandr"], capture_output=True, text=True,
                         timeout=2).stdout
    m = re.search(r"current\s+(\d+)\s*x\s*(\d+)", out)
    if not m:
        raise RuntimeError("no geometry in xrandr output")
    return int(m.group(1)), int(m.group(2))


def screen_px():
    """Screen size in logical pixels; a modest laptop if it cannot be read.

    Deliberately avoids Tk: instantiating a Tk root alongside the macosx
    backend aborts the process ("Tcl_FindHashEntry on deleted table").
    """
    global _SCREEN_PX
    if _SCREEN_PX is None:
        probe = {"darwin": _screen_px_darwin,
                 "win32": _screen_px_windows}.get(sys.platform, _screen_px_x11)
        try:
            w, h = probe()
            _SCREEN_PX = (w, h) if w > 200 and h > 200 else FALLBACK_SCREEN_PX
        except Exception:
            _SCREEN_PX = FALLBACK_SCREEN_PX
    return _SCREEN_PX


def screen_inches(dpi=None, frac_w=0.95, frac_h=0.88):
    """Usable window size in inches, leaving room for menu bar / dock / title."""
    dpi = dpi or matplotlib.rcParams["figure.dpi"]
    w, h = screen_px()
    return w * frac_w / dpi, h * frac_h / dpi


# ------------------------------------------------------------------ palette --
VAL_CMAP = "RdBu_r"      # blue negative -> red positive
DELTA_CMAP = "PRGn"      # magenta down  -> green up
ATTN_CMAP = "YlOrRd"
UP, DOWN = "#1a9850", "#c0392b"
INK = "#1b1b1b"
MUTED = "#7a7a7a"
ACCENT = "#d97706"


# -------------------------------------------------------------------- model --
class Head:
    """One self-attention head. Only W_Q, W_K, W_V are trainable."""

    def __init__(self, sentence, d_model, d_k, seed, lr):
        torch.manual_seed(seed)
        self.sentence_display = sentence
        self.words = sentence.split()
        self.N = len(self.words)
        self.d_model, self.d_k = d_model, d_k
        self.vocab = sorted(set(self.words))
        self.ids = torch.tensor([self.vocab.index(w) for w in self.words])

        # frozen
        self.EMB = torch.randn(len(self.vocab), d_model) * 0.8
        self.POS = torch.randn(self.N, d_model) * 0.30
        self.X = self.EMB[self.ids] + self.POS
        self.W_O = torch.randn(d_k, d_model) * 0.80

        # trainable
        self.W_Q = (torch.randn(d_model, d_k) * 0.40).requires_grad_()
        self.W_K = (torch.randn(d_model, d_k) * 0.40).requires_grad_()
        self.W_V = (torch.randn(d_model, d_k) * 0.70).requires_grad_()

        self.mask = torch.triu(torch.ones(self.N, self.N, dtype=torch.bool), 1)
        self.lr = lr
        self.grads = None
        self.adam_trace = None
        self.grad_loss = None
        self.reset_optimizer()

    def reset_optimizer(self):
        """New training objective -> stale Adam moments would fight it."""
        self.opt = torch.optim.Adam([self.W_Q, self.W_K, self.W_V], lr=self.lr)

    def forward(self):
        Q = self.X @ self.W_Q
        K = self.X @ self.W_K
        V = self.X @ self.W_V
        scores = (Q @ K.T) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask, float("-inf"))
        A = torch.softmax(scores, dim=-1)
        ctx = A @ V
        logits = ctx @ self.W_O @ self.EMB.T
        return Q, K, V, scores, A, ctx, logits

    def loss_at(self, positions, logits):
        pos = torch.tensor(positions)
        return F.cross_entropy(logits[pos], self.ids[pos + 1])

    def snapshot(self, positions):
        with torch.no_grad():
            Q, K, V, scores, A, ctx, logits = self.forward()
            loss = float(self.loss_at(positions, logits))
        return {
            "W_Q": self.W_Q.detach().clone().numpy(),
            "W_K": self.W_K.detach().clone().numpy(),
            "W_V": self.W_V.detach().clone().numpy(),
            "Q": Q.numpy(), "K": K.numpy(), "V": V.numpy(),
            "scores": scores.numpy(), "A": A.numpy(), "ctx": ctx.numpy(),
            "probs": torch.softmax(logits, -1).numpy(),
            "logits": logits.numpy(), "loss": loss,
        }

    def named(self):
        return (("W_Q", self.W_Q), ("W_K", self.W_K), ("W_V", self.W_V))

    def grads_now(self, positions):
        """Gradient at the current weights, without taking a step."""
        for _, prm in self.named():
            prm.grad = None
        *_, logits = self.forward()
        loss = self.loss_at(positions, logits)
        loss.backward()
        g = {n: prm.grad.detach().clone().numpy() for n, prm in self.named()}
        for _, prm in self.named():
            prm.grad = None
        return g, float(loss.detach())

    def train_step(self, positions):
        self.opt.zero_grad()
        *_, logits = self.forward()
        loss = self.loss_at(positions, logits)
        loss.backward()
        self.grads = {n: prm.grad.detach().clone().numpy()
                      for n, prm in self.named()}
        before = {n: prm.detach().clone() for n, prm in self.named()}
        self.grad_loss = float(loss.detach())
        self.opt.step()
        self.adam_trace = self._trace(before)

    def _trace(self, before):
        """Follow the single largest-gradient weight through Adam, exactly."""
        name, gi, gj, gbest = None, 0, 0, -1.0
        for n, _ in self.named():
            g = self.grads[n]
            idx = np.unravel_index(np.abs(g).argmax(), g.shape)
            if abs(g[idx]) > abs(gbest) or name is None:
                name, gi, gj, gbest = n, int(idx[0]), int(idx[1]), float(g[idx])
        prm = dict(self.named())[name]
        st = self.opt.state[prm]
        grp = self.opt.param_groups[0]
        b1, b2 = grp["betas"]
        eps = grp["eps"]
        t = int(float(st["step"]))
        m = float(st["exp_avg"][gi, gj])
        v = float(st["exp_avg_sq"][gi, gj])
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        w0 = float(before[name][gi, gj])
        w1 = float(prm.detach()[gi, gj])
        return dict(name=name, i=gi, j=gj, g=gbest, m=m, v=v, mhat=mhat,
                    vhat=vhat, t=t, b1=b1, b2=b2, eps=eps, lr=self.lr,
                    delta_pred=-self.lr * mhat / (math.sqrt(vhat) + eps),
                    delta_obs=w1 - w0, w0=w0, w1=w1)

    def correct(self, snap, positions):
        pred = snap["logits"][positions].argmax(-1)
        truth = self.ids[torch.tensor(positions) + 1].numpy()
        return int((pred == truth).sum())


# ------------------------------------------------------------------- panels --
def _decimals(mx):
    """Enough decimals to keep ~3 significant digits at the panel's scale."""
    if mx >= 0.5:
        return 2
    if mx >= 0.05:
        return 3
    if mx >= 0.005:
        return 4
    return 5


def _txt_color(x, lim):
    return "white" if lim > 0 and abs(x) / lim > 0.62 else INK


def draw_matrix(ax, M, ref, row_labels, col_labels, title, color_mode,
                highlight_row=None, drift=None, dim_rows=(), annot_size=7.5):
    """Heatmap of M. Cells carry the value plus a ▲/▼ for the change vs ref."""
    M = np.asarray(M, dtype=float)
    D = M - np.asarray(ref, dtype=float)
    disp, cmap = (M, VAL_CMAP) if color_mode == "value" else (D, DELTA_CMAP)
    lim = max(float(np.abs(disp).max()), 1e-6)
    ax.imshow(disp, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")

    dmax = max(float(np.abs(D).max()), 1e-9)
    dec = _decimals(lim)
    size = annot_size - (0.8 if dec >= 4 else 0.0)
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            v, dv = M[i, j], D[i, j]
            shown = v if color_mode == "value" else dv
            ax.text(j, i, f"{shown:+.{dec}f}", ha="center", va="center",
                    fontsize=fs(size), family="monospace",
                    color=_txt_color(disp[i, j], lim))
            if abs(dv) > 5e-4:
                ax.text(j + 0.40, i - 0.32, "▲" if dv > 0 else "▼",
                        ha="right", va="center",
                        fontsize=fs(5.5 + 3.0 * min(1.0, abs(dv) / dmax)),
                        color=UP if dv > 0 else DOWN)

    for r in dim_rows:
        ax.add_patch(Rectangle((-0.5, r - 0.5), cols, 1, facecolor="white",
                               alpha=0.68, ec="none", zorder=4))

    ax.set_xticks(range(cols), col_labels, fontsize=fs(7), color=MUTED)
    ax.set_yticks(range(rows), row_labels, fontsize=fs(7.5), color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)

    if highlight_row is not None:
        ax.add_patch(Rectangle((-0.5, highlight_row - 0.5), cols, 1,
                               fill=False, ec=ACCENT, lw=2.2, zorder=6))
    sub = f"    max Δ since init {drift:.2f}" if drift is not None else ""
    ax.set_title(title + sub, fontsize=fs(9.5), color=INK, pad=fs(18),
                 loc="left")


def draw_attention_matrix(ax, A, refA, words, focus=None):
    A = np.asarray(A)
    shown = np.where(np.triu(np.ones_like(A), 1) > 0, np.nan, A)
    cmap = plt.get_cmap(ATTN_CMAP).copy()
    cmap.set_bad("#f2f2f2")
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    N = len(words)
    D = A - np.asarray(refA)
    for i in range(N):
        for j in range(i + 1):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=fs(6.8), family="monospace",
                    color="white" if A[i, j] > 0.62 else INK)
            if abs(D[i, j]) > 0.01:
                ax.text(j + 0.42, i - 0.30, "▲" if D[i, j] > 0 else "▼",
                        ha="right", va="center", fontsize=fs(6),
                        color=UP if D[i, j] > 0 else DOWN)
    ax.set_xticks(range(N), words, fontsize=fs(7), rotation=45,
                  ha="left", color=MUTED)
    ax.set_yticks(range(N), words, fontsize=fs(7.5), color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    if focus is not None:
        ax.add_patch(Rectangle((-0.5, focus - 0.5), N, 1, fill=False,
                               ec=ACCENT, lw=2.2, zorder=5))
    ax.set_title("attention  A = softmax(QKᵀ/√d_k)     "
                 "row = query word, col = key word",
                 fontsize=fs(9.5), color=INK, pad=fs(32), loc="left")


def draw_attention_row(ax, A, refA, words, focus):
    """One word's attention as bars over the words it is allowed to see."""
    n = focus + 1
    vals, prev = A[focus, :n], refA[focus, :n]
    y = np.arange(n)
    ax.barh(y, prev, color="#d9d9d9", height=0.62, zorder=1)
    ax.barh(y, vals, color=plt.get_cmap(ATTN_CMAP)(0.25 + 0.7 * vals),
            height=0.62, edgecolor="white", zorder=2)
    for i, (v, p) in enumerate(zip(vals, prev)):
        ax.text(min(v + 0.02, 0.9), i, f"{v:.2f}", va="center", fontsize=fs(7.5),
                family="monospace", color=INK, zorder=3)
        if abs(v - p) > 0.005:
            ax.text(1.02, i, "▲" if v > p else "▼", va="center", fontsize=fs(8),
                    color=UP if v > p else DOWN)
    ax.set_yticks(y, words[:n], fontsize=fs(8))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"], fontsize=fs(7), color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f'what "{words[focus]}" attends to      '
                 f"(grey bar = previous step)",
                 fontsize=fs(9.5), color=INK, pad=fs(5), loc="left")


def draw_loss(ax, xs, losses, scope_label):
    ax.plot(xs, losses, color=ACCENT, lw=1.8)
    ax.scatter([xs[-1]], [losses[-1]], s=26, color=ACCENT, zorder=3)
    ax.set_xlabel("training step", fontsize=fs(8), color=MUTED)
    if len(xs) == 1:
        ax.set_xlim(xs[0] - 0.5, xs[0] + 1.0)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"loss  ({scope_label})    {losses[0]:.3f} → {losses[-1]:.3f}",
                 fontsize=fs(9.5), color=INK, pad=fs(5), loc="left")


def draw_predictions_all(ax, head, snap, positions):
    """Probability given to the true next word, per position."""
    p = [snap["probs"][i, int(head.ids[i + 1])] for i in positions]
    ok = [snap["logits"][i].argmax() == int(head.ids[i + 1]) for i in positions]
    labels = [f"{head.words[i]} → {head.words[i + 1]}" for i in positions]
    y = np.arange(len(positions))
    ax.barh(y, p, color=[UP if o else "#e8b4ae" for o in ok], height=0.62)
    for i, v in enumerate(p):
        ax.text(min(v + 0.015, 0.88), i, f"{v:.2f}", va="center",
                fontsize=fs(7.5), family="monospace", color=INK)
    ax.set_yticks(y, labels, fontsize=fs(7.5))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"], fontsize=fs(7), color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"P(true next word)     {sum(ok)}/{len(positions)} correct",
                 fontsize=fs(9.5), color=INK, pad=fs(5), loc="left")


def draw_prediction_one(ax, head, snap, focus):
    probs = snap["probs"][focus]
    truth = int(head.ids[focus + 1])
    x = np.arange(len(head.vocab))
    ax.bar(x, probs, color=["#cfcfcf"] * len(x), width=0.68)
    ax.patches[truth].set_facecolor(UP if probs.argmax() == truth else "#e8b4ae")
    ax.set_xticks(x, head.vocab, fontsize=fs(7.5), rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0], ["0", ".5", "1"], fontsize=fs(7), color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f'after "{head.words[focus]}" it predicts '
                 f'"{head.vocab[int(probs.argmax())]}"   '
                 f'(truth "{head.words[focus + 1]}", p={probs[truth]:.2f})',
                 fontsize=fs(9.5), color=INK, pad=fs(5), loc="left")


def draw_text_panel(ax, title, rows, mono=False, fontsize=fs(9.3), label_w=0.105,
                    wrap=185):
    """rows: (label, text). label "" starts at the margin, None indents as
    a wrapped continuation of the line above."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, fontsize=fs(9.5), color=INK, pad=fs(4), loc="left")

    if not mono:
        flowed = []
        for label, text in rows:
            parts = textwrap.wrap(text, width=wrap) or [""]
            flowed.append((label, parts[0]))
            flowed.extend((None, extra) for extra in parts[1:])
        rows = flowed

    fam = "monospace" if mono else "sans-serif"
    n = max(len(rows), 1)
    for k, (label, text) in enumerate(rows):
        y = 1.0 - (k + 0.6) / n
        if label:
            ax.text(0, y, label, fontsize=fs(fontsize), color=ACCENT,
                    family=fam, va="center", weight="bold")
        ax.text(0 if label == "" else label_w, y, text, fontsize=fs(fontsize),
                color=INK, family=fam, va="center")


def narrate(app):
    """Plain-language reading of what the last training step actually did."""
    head, cur, prev = app.head, app.cur, app.prev
    words, N = head.words, head.N

    fixed = ("the idea",
             "Q and K decide WHERE each word looks; V decides WHAT gets copied "
             "back. The W matrices are shared by every word, so one weight cell "
             "moves a whole column of Q, K or V at once.")
    if app.step == 0:
        return [
            ("start", f"Nothing trained yet. W_Q, W_K, W_V are random draws, so "
                      f"every word's attention is close to an even split over the "
                      f"words it can see, and the loss sits near "
                      f"ln({len(head.vocab)}) = {math.log(len(head.vocab)):.2f}."),
            fixed,
            ("next", "Press Enter. The ▲▼ marks then show what one gradient step "
                     "did to each number."),
        ]

    rows = []
    dl = cur["loss"] - prev["loss"]
    verb = "fell" if dl < 0 else "rose"
    rows.append(("loss", f"{prev['loss']:.4f} → {cur['loss']:.4f} ({dl:+.4f}) — "
                         f"it {verb} on the {len(app.positions)} position"
                         f"{'s' if len(app.positions) > 1 else ''} being trained."))

    # biggest weight move, and what it did downstream
    bw, bi, bj, bd = None, 0, 0, 0.0
    for n in ("W_Q", "W_K", "W_V"):
        D = cur[n] - prev[n]
        idx = np.unravel_index(np.abs(D).argmax(), D.shape)
        if bw is None or abs(D[idx]) > abs(bd):
            bw, bi, bj, bd = n, int(idx[0]), int(idx[1]), float(D[idx])
    out = {"W_Q": "Q", "W_K": "K", "W_V": "V"}[bw]
    colname = f"{out.lower()}{bj}"
    colmove = float(np.abs(cur[out][:, bj] - prev[out][:, bj]).mean())
    rows.append((
        "weights",
        f"the largest move was {bw}[d{bi},{colname}] {bd:+.4f}. Because "
        f"{out} = X · {bw}, that single number shifted column {colname} of {out} "
        f"for all {N} words (mean |Δ| there {colmove:.4f})."))

    # biggest attention move among the trained rows, traced back to the score
    A, Ap, S, Sp = cur["A"], prev["A"], cur["scores"], prev["scores"]
    bi2 = bj2 = 0
    bd2 = None
    for i in app.positions:
        for j in range(i + 1):
            d = float(A[i, j] - Ap[i, j])
            if bd2 is None or abs(d) > abs(bd2):
                bi2, bj2, bd2 = i, j, d
    others = [j for j in range(bi2 + 1) if j != bj2]
    if others:
        rival = float(np.mean([S[bi2, j] - Sp[bi2, j] for j in others]))
        rival_txt = (f", while the {len(others)} competing score"
                     f"{'s' if len(others) > 1 else ''} in that row moved "
                     f"{rival:+.3f} on average")
    else:
        rival_txt = " (nothing to compete with, so this row is pinned at 1.00)"
    rows.append((
        "attention",
        f'"{words[bi2]}" → "{words[bj2]}" went {Ap[bi2, bj2]:.3f} → '
        f"{A[bi2, bj2]:.3f} ({bd2:+.3f}). Its raw score q·k/√d_k moved "
        f"{Sp[bi2, bj2]:+.3f} → {S[bi2, bj2]:+.3f}{rival_txt} — softmax turns "
        f"the GAPS within a row into probabilities, so a score can fall and its "
        f"attention still rise."))

    # exact split of the context change: attention moving vs V moving
    n_seen = bi2 + 1
    dA_part = (A[bi2, :n_seen] - Ap[bi2, :n_seen]) @ prev["V"][:n_seen]
    dV_part = Ap[bi2, :n_seen] @ (cur["V"][:n_seen] - prev["V"][:n_seen])
    cross = (A[bi2, :n_seen] - Ap[bi2, :n_seen]) @ (cur["V"][:n_seen]
                                                    - prev["V"][:n_seen])
    total = cur["ctx"][bi2] - prev["ctx"][bi2]
    rows.append((
        "what it read",
        f'the vector "{words[bi2]}" reads out moved by ‖Δ‖='
        f"{np.linalg.norm(total):.3f}. As vectors it splits exactly: "
        f"Δctx = ΔA·V + A·ΔV + ΔA·ΔV, with magnitudes "
        f"{np.linalg.norm(dA_part):.3f} (looked elsewhere), "
        f"{np.linalg.norm(dV_part):.3f} (V itself moved), "
        f"{np.linalg.norm(cross):.3f} (cross) — those three do not add up to "
        f"‖Δ‖ because they point in different directions."))

    rows.append(fixed)
    return rows


def backprop_rows(app):
    """The exact numbers behind the step, for short sentences."""
    head = app.head
    P = app.positions
    T = len(P)
    snap = app.prev if app.step > 0 else app.cur
    probs, vocab = snap["probs"], head.vocab
    dk, dm, N, nv = head.d_k, head.d_model, head.N, len(vocab)
    rows = []

    when = (f"the weights before step {app.step}" if app.step > 0
            else "the current weights — this is what the first step will use")
    rows.append(("forward", f"loss and gradients computed at {when}."))
    terms = []
    for i in P:
        tgt = int(head.ids[i + 1])
        pt = float(probs[i, tgt])
        terms.append(-math.log(max(pt, 1e-12)))
        rows.append(("", f'   "{head.words[i]}" → "{head.words[i + 1]}"'
                         f"      p = {pt:.4f}      −log p = {terms[-1]:.4f}"))
    joined = " + ".join(f"{t:.4f}" for t in terms)
    rows.append(("loss", f"L = mean(−log p) = ({joined}) / {T} = "
                         f"{sum(terms) / T:.6f}"))

    i0 = P[0]
    g0 = probs[i0].copy()
    g0[int(head.ids[i0 + 1])] -= 1.0
    g0 /= T
    vec = ", ".join(f"{x:+.3f}" for x in g0)
    rows.append(("backward", f"dL/dlogits[{head.words[i0]}] = (softmax − onehot)/T "
                             f"= [{vec}]   over vocab ({', '.join(vocab)})"))
    for line in (
        f"dL/dctx = dL/dlogits · (W_O·Embᵀ)ᵀ      [{T},{nv}]·[{nv},{dk}] → [{T},{dk}]",
        f"dL/dV   = Aᵀ · dL/dctx                                    → [{N},{dk}]",
        f"dL/dA   = dL/dctx · Vᵀ                                    → [{T},{N}]",
        f"dL/dS   = A ⊙ (dL/dA − rowsum(dL/dA ⊙ A))   softmax jac   → [{T},{N}]",
        f"dL/dQ   = (dL/dS · K)/√d_k        dL/dK = (dL/dSᵀ · Q)/√d_k → [{N},{dk}]",
        f"dL/dW_Q = Xᵀ · dL/dQ    (and likewise for W_K, W_V)       → [{dm},{dk}]",
    ):
        rows.append(("", "   " + line))

    tr = head.adam_trace
    if tr is None:
        rows.append(("adam", "take a step to see the update arithmetic."))
        return rows
    pre = {"W_Q": "q", "W_K": "k", "W_V": "v"}[tr["name"]]
    cell = f"{tr['name']}[d{tr['i']},{pre}{tr['j']}]"
    diff = abs(tr["delta_pred"] - tr["delta_obs"])
    ok = "✓" if diff < 1e-7 else f"MISMATCH ({diff:.2e})"
    rows.append(("adam", f"largest-gradient cell {cell}, optimiser step t={tr['t']}"
                         f"   (β1={tr['b1']}, β2={tr['b2']}, lr={tr['lr']})"))
    rows.append(("", f"   g = {tr['g']:+.6f}"))
    rows.append(("", f"   m = β1·m + (1−β1)·g = {tr['m']:+.6f}"
                     f"     m̂ = m/(1−β1^t) = {tr['mhat']:+.6f}"))
    rows.append(("", f"   v = β2·v + (1−β2)·g² = {tr['v']:.8f}"
                     f"   v̂ = v/(1−β2^t) = {tr['vhat']:.8f}"))
    rows.append(("", f"   Δ = −lr·m̂/(√v̂+ε) = −{tr['lr']}·({tr['mhat']:+.6f})/"
                     f"({math.sqrt(tr['vhat']):.6f}+{tr['eps']:g}) = "
                     f"{tr['delta_pred']:+.6f}"))
    rows.append(("", f"   {cell}: {tr['w0']:+.6f} → {tr['w1']:+.6f}"
                     f"   observed Δ {tr['delta_obs']:+.6f}  {ok}"))
    return rows


# ---------------------------------------------------------------- the app ----
class App:
    def __init__(self, head, scope, focus):
        self.head = head
        self.scope = scope            # "word" | "sentence"
        self.focus = focus
        self.color_mode = "value"     # "value" | "delta"
        self.ref_mode = "prev"        # "prev"  | "init"
        self.step = 0

        self.base = head.snapshot(self.positions)
        self.prev = self.base
        self.cur = self.base
        self.losses = [self.base["loss"]]
        self.loss_x = [0]

        # short sentences get the exact backprop arithmetic underneath
        self.show_math = head.N < 5
        self.pending_grads = None

        # never open a window taller than the screen, or the controls fall off
        self.math_available = self.show_math
        if self.show_math and self._fit_scale(True) < LEGIBLE_FS:
            self.show_math = False      # [b] brings it back on demand
        size = self._fit()
        self.fig = plt.figure(figsize=size)
        self.fig.canvas.manager.set_window_title("Q, K, V during training")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.build_layout()
        self.draw()

    # -- fitting ---------------------------------------------------------------
    @staticmethod
    def _want(show_math):
        return 16.5, 15.4 if show_math else 10.9

    def _fit_scale(self, show_math):
        want_w, want_h = self._want(show_math)
        max_w, max_h = screen_inches()
        return min(1.0, max_w / want_w, max_h / want_h) * FIG_SCALE

    def _fit(self):
        """Set the global font scale and return the figure size to use."""
        global FS
        FS = self._fit_scale(self.show_math)
        want_w, want_h = self._want(self.show_math)
        return want_w * FS, want_h * FS

    # -- state ---------------------------------------------------------------
    @property
    def positions(self):
        if self.scope == "word":
            return [self.focus]
        return list(range(self.head.N - 1))

    @property
    def scope_label(self):
        if self.scope == "word":
            return f'single word: "{self.head.words[self.focus]}"'
        return f"full sentence: all {self.head.N - 1} positions"

    def ref(self):
        return self.prev if self.ref_mode == "prev" else self.base

    def resync(self):
        """The objective changed: restart the loss curve and Adam's state."""
        self.head.reset_optimizer()
        self.cur = self.head.snapshot(self.positions)
        self.prev = self.cur
        self.losses = [self.cur["loss"]]
        self.loss_x = [self.step]
        self.pending_grads = None

    # -- layout --------------------------------------------------------------
    def build_layout(self):
        self.fig.clear()
        if self.show_math:
            nrows = 6
            ratios = [0.85, 0.95, 1.05, 1.0, 0.85, 1.85]
            top, bottom, head_y, sub_y = 0.895, 0.048, 0.972, 0.950
        else:
            nrows = 4
            ratios = [1.0, 1.45, 1.2, 0.95]
            top, bottom, head_y, sub_y = 0.862, 0.075, 0.972, 0.944
        gs = gridspec.GridSpec(
            nrows, 3, figure=self.fig, height_ratios=ratios,
            left=0.055, right=0.985, top=top, bottom=bottom,
            hspace=0.55, wspace=0.20)
        self.ax_w = [self.fig.add_subplot(gs[0, c]) for c in range(3)]
        self.ax_p = [self.fig.add_subplot(gs[1, c]) for c in range(3)]
        self.ax_b = [self.fig.add_subplot(gs[2, c]) for c in range(3)]
        self.ax_note = self.fig.add_subplot(gs[3, :])
        if self.show_math:
            self.ax_g = [self.fig.add_subplot(gs[4, c]) for c in range(3)]
            self.ax_math = self.fig.add_subplot(gs[5, :])
        else:
            self.ax_g, self.ax_math = [], None

        self.subtitle = self.fig.text(0.055, sub_y, "", fontsize=fs(9), color=MUTED)
        self._head_y = head_y
        hint = ("[Enter] one step    [r] ten steps    "
                "[d] Δ vs previous / vs init    [v] colour by value / by Δ    "
                "[m] switch scope    [←/→] pick word")
        if self.math_available:
            hint += ("    [b] hide backprop block" if self.show_math
                     else "    [b] SHOW backprop block")
        self.fig.text(0.055, 0.012 if self.show_math else 0.022,
                      hint + "    [q] quit", fontsize=fs(8.5), color=MUTED)

        # kept in the header: a resized window can never push these off-screen
        bh = 0.030 * (10.9 / (15.4 if self.show_math else 10.9))
        by = head_y - bh * 0.55
        bax = self.fig.add_axes([0.845, by, 0.06, bh])
        self.btn_next = Button(bax, "Next  ▶", color="#efefef", hovercolor="#e0e0e0")
        self.btn_next.label.set_fontsize(fs(9))
        self.btn_next.on_clicked(lambda _evt: self.advance(1))

        bax10 = self.fig.add_axes([0.915, by, 0.06, bh])
        self.btn_ten = Button(bax10, "×10  ▶▶", color="#efefef", hovercolor="#e0e0e0")
        self.btn_ten.label.set_fontsize(fs(9))
        self.btn_ten.on_clicked(lambda _evt: self.advance(10))

    # -- events --------------------------------------------------------------
    def on_key(self, event):
        k = event.key
        if k in ("enter", " "):
            self.advance(1)
        elif k == "r":
            self.advance(10)
        elif k == "d":
            self.ref_mode = "init" if self.ref_mode == "prev" else "prev"
            self.draw()
        elif k == "v":
            self.color_mode = "delta" if self.color_mode == "value" else "value"
            self.draw()
        elif k == "m":
            self.scope = "sentence" if self.scope == "word" else "word"
            self.resync()
            self.build_layout()
            self.draw()
        elif k in ("left", "right") and self.scope == "word":
            n = self.head.N - 1
            self.focus = (self.focus + (1 if k == "right" else -1)) % n
            self.resync()
            self.draw()
        elif k == "b" and self.math_available:
            self.show_math = not self.show_math
            self.fig.set_size_inches(*self._fit(), forward=True)
            self.build_layout()
            self.draw()
        elif k == "q":
            plt.close(self.fig)

    def advance(self, n):
        for _ in range(n):
            self.head.train_step(self.positions)
            self.step += 1
            self.prev = self.cur
            self.cur = self.head.snapshot(self.positions)
            self.losses.append(self.cur["loss"])
            self.loss_x.append(self.step)
        self.draw()

    # -- render --------------------------------------------------------------
    def draw(self):
        head, cur, ref = self.head, self.cur, self.ref()
        for ax in self.ax_w + self.ax_p + self.ax_b + self.ax_g:
            ax.cla()
        self.ax_note.cla()
        if self.ax_math is not None:
            self.ax_math.cla()

        dims = [f"d{i}" for i in range(head.d_model)]
        for ax, name, pre in zip(self.ax_w, ("W_Q", "W_K", "W_V"), "qkv"):
            cols = [f"{pre}{i}" for i in range(head.d_k)]
            draw_matrix(ax, cur[name], ref[name], dims, cols, name,
                        self.color_mode,
                        drift=float(np.abs(cur[name] - self.base[name]).max()))

        f, N = self.focus, head.N
        word_scope = self.scope == "word"
        titles = ("Q = X · W_Q   (queries)", "K = X · W_K   (keys)",
                  "V = X · W_V   (values)")
        # in single-word scope, only Q[focus] and K/V rows 0..focus are used
        dims_off = {
            "Q": [i for i in range(N) if i != f] if word_scope else [],
            "K": list(range(f + 1, N)) if word_scope else [],
            "V": list(range(f + 1, N)) if word_scope else [],
        }
        for ax, name, title, pre in zip(self.ax_p, ("Q", "K", "V"), titles, "qkv"):
            cols = [f"{pre}{i}" for i in range(head.d_k)]
            draw_matrix(ax, cur[name], ref[name], head.words, cols, title,
                        self.color_mode,
                        highlight_row=f if (word_scope and name == "Q") else None,
                        dim_rows=dims_off[name])

        if self.scope == "word":
            draw_attention_row(self.ax_b[0], cur["A"], ref["A"],
                               head.words, self.focus)
            draw_loss(self.ax_b[1], self.loss_x, self.losses, self.scope_label)
            draw_prediction_one(self.ax_b[2], head, cur, self.focus)
        else:
            draw_attention_matrix(self.ax_b[0], cur["A"], ref["A"], head.words)
            draw_loss(self.ax_b[1], self.loss_x, self.losses, self.scope_label)
            draw_predictions_all(self.ax_b[2], head, cur, self.positions)

        draw_text_panel(
            self.ax_note,
            "what the last step did     (always measured against the previous "
            "step, whatever [d] is set to)",
            narrate(self))

        if self.show_math:
            if self.step > 0:
                grads = head.grads
                gtitle = f"gradients used by step {self.step}"
            else:
                if self.pending_grads is None:
                    self.pending_grads, _ = head.grads_now(self.positions)
                grads = self.pending_grads
                gtitle = "gradients at the current weights (step 1 will use these)"
            dims_g = [f"d{i}" for i in range(head.d_model)]
            for ax, name, pre in zip(self.ax_g, ("W_Q", "W_K", "W_V"), "qkv"):
                draw_matrix(ax, grads[name], grads[name],
                            dims_g, [f"{pre}{i}" for i in range(head.d_k)],
                            f"∂L/∂{name}", "value", annot_size=7.0)
            self.ax_g[1].set_title(f"∂L/∂W_K        {gtitle}", fontsize=fs(9.5),
                                   color=INK, pad=fs(18), loc="left")
            draw_text_panel(self.ax_math,
                            "the exact arithmetic behind this step",
                            backprop_rows(self), mono=True, fontsize=fs(8.6),
                            label_w=0.075)

        dl = cur["loss"] - self.prev["loss"]
        ok = head.correct(cur, self.positions)
        cmp_txt = "previous step" if self.ref_mode == "prev" else "step 0 (init)"
        self.fig.suptitle(
            f'"{head.sentence_display}"          '
            f"step {self.step}   ·   loss {cur['loss']:.4f} ({dl:+.4f})   ·   "
            f"{ok}/{len(self.positions)} next words correct",
            fontsize=fs(13), color=INK, x=0.055, ha="left", y=self._head_y)
        self.subtitle.set_text(
            f"scope — {self.scope_label}    |    trainable: W_Q, W_K, W_V "
            f"(everything else frozen)    |    ▲▼ vs {cmp_txt}    |    "
            f"cells coloured by {'value' if self.color_mode == 'value' else 'change'}")
        self.fig.canvas.draw_idle()


# ------------------------------------------------------------------- menu ----
def choose(head, scope, word):
    if scope is None:
        print("\n  Scope")
        print("    1  single word    train + watch ONE word predicting the next")
        print(f"    2  full sentence  train + watch all {head.N - 1} positions")
        while True:
            c = input("\n  choose 1 or 2 [1]: ").strip() or "1"
            if c in ("1", "2"):
                scope = "word" if c == "1" else "sentence"
                break
    if scope == "sentence":
        # a sensible focus to land on if the user later presses [m]
        return scope, head.N - 2

    options = list(range(head.N - 1))    # last word has no "next word"
    if word is not None:
        if word.isdigit() and int(word) in options:
            return scope, int(word)
        if word in head.words and head.words.index(word) in options:
            i = head.words.index(word)
            if head.words.count(word) > 1:
                print(f"  ! '{word}' appears {head.words.count(word)} times, "
                      f"using position {i}")
            return scope, i
        print(f"  ! '{word}' is not a usable query word, pick from the list")

    print("\n  Which word is the query?")
    for i in options:
        note = "  (position 0 can only see itself — Q·K has no effect there)" \
            if i == 0 else f'  → must predict "{head.words[i + 1]}"'
        print(f"    {i}  {head.words[i]:<10}{note}")
    while True:
        c = input(f"\n  choose 0-{options[-1]} [{options[-1]}]: ").strip()
        c = c or str(options[-1])
        if c.isdigit() and int(c) in options:
            return scope, int(c)


DEFAULT_SENTENCE = "the hungry cat quietly chased a small brown mouse"


def normalise(text):
    """Lowercase, drop punctuation. Keeps 'Cat,' and 'cat' as one vocab entry."""
    out = []
    for tok in text.split():
        w = tok.strip(".,;:!?()[]\"'`“”‘’—–").lower()
        if w:
            out.append(w)
    return out


def ask_sentence(default):
    print("\n  Sentence to train on")
    print(f'    default: "{default}"')
    print("    8-10 words fits the window best; punctuation and capitals are "
          "stripped.")
    while True:
        raw = input("\n  sentence [Enter for the default]: ").strip()
        if not raw:
            return default
        words = normalise(raw)
        if len(words) < 3:
            print("  ! need at least 3 real words")
            continue
        if len(words) > 14:
            print(f"  ! {len(words)} words — the heatmaps will be cramped")
            if input("  use it anyway? [y/N]: ").strip().lower() != "y":
                continue
        return " ".join(words)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sentence", default=None,
                   help="skip the sentence prompt")
    p.add_argument("--scope", choices=("word", "sentence"), default=None,
                   help="skip the scope menu")
    p.add_argument("--word", default=None,
                   help="query word (text or index) for --scope word")
    p.add_argument("--d-model", type=int, default=5)
    p.add_argument("--d-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--fig-scale", type=float, default=1.0,
                   help="nudge the auto-fitted window size (e.g. 0.9, 1.15)")
    p.add_argument("--screen", default=None, metavar="WxH",
                   help="override screen detection, e.g. 1680x1050")
    args = p.parse_args()

    global FIG_SCALE, _SCREEN_PX
    FIG_SCALE = args.fig_scale
    if args.screen:
        try:
            w, h = (int(v) for v in args.screen.lower().split("x"))
            _SCREEN_PX = (w, h)
        except ValueError:
            sys.exit("  ! --screen wants WxH, e.g. 1680x1050")

    if args.sentence is None:
        sentence = ask_sentence(DEFAULT_SENTENCE)
    else:
        words = normalise(args.sentence)
        if len(words) < 3:
            sys.exit("  ! --sentence needs at least 3 words")
        sentence = " ".join(words)

    head = Head(sentence, args.d_model, args.d_k, args.seed, args.lr)
    print(f'\n  sentence: "{sentence}"')
    print(f"  {head.N} words, {len(head.vocab)} unique (that is the whole "
          f"vocabulary), {head.N - 1} next-word predictions to learn")
    scope, focus = choose(head, args.scope, args.word)
    print(f"\n  opening the plot window — press Enter in it for one training "
          f"step.\n  (the window must have focus for keys to register)\n")

    App(head, scope, focus)
    plt.show()


if __name__ == "__main__":
    main()
