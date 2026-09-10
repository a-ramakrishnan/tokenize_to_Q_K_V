#!/usr/bin/env python3
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
because of those three matrices. No backprop internals are drawn, only the
before/after state of Q, K and V.

The sentence, the scope and the query word are all asked for at startup;
any of them can be supplied as a flag to skip that prompt.

  python3 qkv_plot_viz.py
  python3 qkv_plot_viz.py --sentence "the dog chased the ball across the park"
  python3 qkv_plot_viz.py --scope word --word brown
"""

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

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

    def train_step(self, positions):
        self.opt.zero_grad()
        *_, logits = self.forward()
        self.loss_at(positions, logits).backward()
        self.opt.step()

    def correct(self, snap, positions):
        pred = snap["logits"][positions].argmax(-1)
        truth = self.ids[torch.tensor(positions) + 1].numpy()
        return int((pred == truth).sum())


# ------------------------------------------------------------------- panels --
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
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            v, dv = M[i, j], D[i, j]
            shown = v if color_mode == "value" else dv
            ax.text(j, i, f"{shown:+.2f}", ha="center", va="center",
                    fontsize=annot_size, family="monospace",
                    color=_txt_color(disp[i, j], lim))
            if abs(dv) > 5e-4:
                ax.text(j + 0.40, i - 0.32, "▲" if dv > 0 else "▼",
                        ha="right", va="center",
                        fontsize=5.5 + 3.0 * min(1.0, abs(dv) / dmax),
                        color=UP if dv > 0 else DOWN)

    for r in dim_rows:
        ax.add_patch(Rectangle((-0.5, r - 0.5), cols, 1, facecolor="white",
                               alpha=0.68, ec="none", zorder=4))

    ax.set_xticks(range(cols), col_labels, fontsize=7, color=MUTED)
    ax.set_yticks(range(rows), row_labels, fontsize=7.5, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)

    if highlight_row is not None:
        ax.add_patch(Rectangle((-0.5, highlight_row - 0.5), cols, 1,
                               fill=False, ec=ACCENT, lw=2.2, zorder=6))
    sub = f"    max Δ since init {drift:.2f}" if drift is not None else ""
    ax.set_title(title + sub, fontsize=9.5, color=INK, pad=18, loc="left")


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
                    fontsize=6.8, family="monospace",
                    color="white" if A[i, j] > 0.62 else INK)
            if abs(D[i, j]) > 0.01:
                ax.text(j + 0.42, i - 0.30, "▲" if D[i, j] > 0 else "▼",
                        ha="right", va="center", fontsize=6,
                        color=UP if D[i, j] > 0 else DOWN)
    ax.set_xticks(range(N), words, fontsize=7, rotation=45,
                  ha="left", color=MUTED)
    ax.set_yticks(range(N), words, fontsize=7.5, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    if focus is not None:
        ax.add_patch(Rectangle((-0.5, focus - 0.5), N, 1, fill=False,
                               ec=ACCENT, lw=2.2, zorder=5))
    ax.set_title("attention  A = softmax(QKᵀ/√d_k)     "
                 "row = query word, col = key word",
                 fontsize=9.5, color=INK, pad=32, loc="left")


def draw_attention_row(ax, A, refA, words, focus):
    """One word's attention as bars over the words it is allowed to see."""
    n = focus + 1
    vals, prev = A[focus, :n], refA[focus, :n]
    y = np.arange(n)
    ax.barh(y, prev, color="#d9d9d9", height=0.62, zorder=1)
    ax.barh(y, vals, color=plt.get_cmap(ATTN_CMAP)(0.25 + 0.7 * vals),
            height=0.62, edgecolor="white", zorder=2)
    for i, (v, p) in enumerate(zip(vals, prev)):
        ax.text(min(v + 0.02, 0.9), i, f"{v:.2f}", va="center", fontsize=7.5,
                family="monospace", color=INK, zorder=3)
        if abs(v - p) > 0.005:
            ax.text(1.02, i, "▲" if v > p else "▼", va="center", fontsize=8,
                    color=UP if v > p else DOWN)
    ax.set_yticks(y, words[:n], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"], fontsize=7, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f'what "{words[focus]}" attends to      '
                 f"(grey bar = previous step)",
                 fontsize=9.5, color=INK, pad=5, loc="left")


def draw_loss(ax, xs, losses, scope_label):
    ax.plot(xs, losses, color=ACCENT, lw=1.8)
    ax.scatter([xs[-1]], [losses[-1]], s=26, color=ACCENT, zorder=3)
    ax.set_xlabel("training step", fontsize=8, color=MUTED)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7, colors=MUTED, length=0)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"loss  ({scope_label})    {losses[0]:.3f} → {losses[-1]:.3f}",
                 fontsize=9.5, color=INK, pad=5, loc="left")


def draw_predictions_all(ax, head, snap, positions):
    """Probability given to the true next word, per position."""
    p = [snap["probs"][i, int(head.ids[i + 1])] for i in positions]
    ok = [snap["logits"][i].argmax() == int(head.ids[i + 1]) for i in positions]
    labels = [f"{head.words[i]} → {head.words[i + 1]}" for i in positions]
    y = np.arange(len(positions))
    ax.barh(y, p, color=[UP if o else "#e8b4ae" for o in ok], height=0.62)
    for i, v in enumerate(p):
        ax.text(min(v + 0.015, 0.88), i, f"{v:.2f}", va="center",
                fontsize=7.5, family="monospace", color=INK)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"], fontsize=7, color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"P(true next word)     {sum(ok)}/{len(positions)} correct",
                 fontsize=9.5, color=INK, pad=5, loc="left")


def draw_prediction_one(ax, head, snap, focus):
    probs = snap["probs"][focus]
    truth = int(head.ids[focus + 1])
    x = np.arange(len(head.vocab))
    ax.bar(x, probs, color=["#cfcfcf"] * len(x), width=0.68)
    ax.patches[truth].set_facecolor(UP if probs.argmax() == truth else "#e8b4ae")
    ax.set_xticks(x, head.vocab, fontsize=7.5, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0], ["0", ".5", "1"], fontsize=7, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f'after "{head.words[focus]}" it predicts '
                 f'"{head.vocab[int(probs.argmax())]}"   '
                 f'(truth "{head.words[focus + 1]}", p={probs[truth]:.2f})',
                 fontsize=9.5, color=INK, pad=5, loc="left")


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

        self.fig = plt.figure(figsize=(16.5, 9.6))
        self.fig.canvas.manager.set_window_title("Q, K, V during training")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.build_layout()
        self.draw()

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

    # -- layout --------------------------------------------------------------
    def build_layout(self):
        self.fig.clear()
        gs = gridspec.GridSpec(
            3, 3, figure=self.fig, height_ratios=[1.0, 1.5, 1.25],
            left=0.055, right=0.985, top=0.845, bottom=0.085,
            hspace=0.42, wspace=0.20)
        self.ax_w = [self.fig.add_subplot(gs[0, c]) for c in range(3)]
        self.ax_p = [self.fig.add_subplot(gs[1, c]) for c in range(3)]
        self.ax_b = [self.fig.add_subplot(gs[2, c]) for c in range(3)]

        self.subtitle = self.fig.text(0.055, 0.938, "", fontsize=9, color=MUTED)
        self.fig.text(0.055, 0.022,
                      "[Enter] one step    [r] ten steps    "
                      "[d] Δ vs previous / vs init    [v] colour by value / by Δ    "
                      "[m] switch scope    [←/→] pick word    [q] quit",
                      fontsize=8.5, color=MUTED)

        bax = self.fig.add_axes([0.845, 0.012, 0.06, 0.032])
        self.btn_next = Button(bax, "Next  ▶", color="#efefef", hovercolor="#e0e0e0")
        self.btn_next.label.set_fontsize(9)
        self.btn_next.on_clicked(lambda _evt: self.advance(1))

        bax10 = self.fig.add_axes([0.915, 0.012, 0.06, 0.032])
        self.btn_ten = Button(bax10, "×10  ▶▶", color="#efefef", hovercolor="#e0e0e0")
        self.btn_ten.label.set_fontsize(9)
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
        for ax in self.ax_w + self.ax_p + self.ax_b:
            ax.cla()

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

        dl = cur["loss"] - self.prev["loss"]
        ok = head.correct(cur, self.positions)
        cmp_txt = "previous step" if self.ref_mode == "prev" else "step 0 (init)"
        self.fig.suptitle(
            f'"{head.sentence_display}"          '
            f"step {self.step}   ·   loss {cur['loss']:.4f} ({dl:+.4f})   ·   "
            f"{ok}/{len(self.positions)} next words correct",
            fontsize=13, color=INK, x=0.055, ha="left", y=0.972)
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
    args = p.parse_args()

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
