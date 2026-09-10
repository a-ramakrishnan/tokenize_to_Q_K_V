#!/usr/bin/env python3
"""
qkv_train_viz.py — watch Q, K and V change while a tiny attention head trains.

Setup
-----
  sentence        : 9 words (any 8-10 word sentence works, --sentence)
  X  (9 x 5)      : frozen embeddings + frozen positional vectors
  W_Q,W_K,W_V     : (5 x 4) random at init  <-- THE ONLY TRAINABLE THINGS
  W_O (4 x 5)     : frozen output projection
  task            : causal next-word prediction, cross-entropy loss

Because everything except W_Q/W_K/W_V is frozen, every number that moves on
screen moved because of those three matrices.

Controls: [Enter] next step   [N] run N steps   [d] delta mode   [q] quit

  python3 qkv_train_viz.py
  python3 qkv_train_viz.py --sentence "the dog chased the ball across the quiet park"
"""

import argparse
import math
import shutil
import sys

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------- config ---
_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--sentence", default=None,
                help="8-10 words works best on one screen; prompts if omitted")
_p.add_argument("--d-model", type=int, default=5, help="embedding width")
_p.add_argument("--d-k", type=int, default=4, help="query/key/value width")
_p.add_argument("--seed", type=int, default=1)
_p.add_argument("--lr", type=float, default=0.06)
_p.add_argument("--max-steps", type=int, default=200)
_args = _p.parse_args()

DEFAULT_SENTENCE = "the hungry cat quietly chased a small brown mouse"


def _normalise(text):
    """Lowercase, drop punctuation, so 'Cat,' and 'cat' are one vocab entry."""
    out = []
    for tok in text.split():
        w = tok.strip(".,;:!?()[]\"'`“”‘’—–").lower()
        if w:
            out.append(w)
    return out


def _ask_sentence(default):
    print("\n  Sentence to train on")
    print(f'    default: "{default}"')
    print("    8-10 words fits one screen; punctuation and capitals are stripped.")
    while True:
        raw = input("\n  sentence [Enter for the default]: ").strip()
        if not raw:
            return default
        got = _normalise(raw)
        if len(got) < 3:
            print("  ! need at least 3 real words")
            continue
        if len(got) > 14:
            print(f"  ! {len(got)} words - the rows will wrap badly")
            if input("  use it anyway? [y/N]: ").strip().lower() != "y":
                continue
        return " ".join(got)


if _args.sentence is None:
    SENTENCE = _ask_sentence(DEFAULT_SENTENCE)
else:
    _got = _normalise(_args.sentence)
    if len(_got) < 3:
        raise SystemExit("  ! --sentence needs at least 3 words")
    SENTENCE = " ".join(_got)
D_MODEL = _args.d_model
D_K = _args.d_k
SEED = _args.seed
LR = _args.lr
MAX_STEPS = _args.max_steps

# ------------------------------------------------------------------ color ---
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def fg(c):
    return f"\033[38;5;{c}m"


# blue (negative) -> grey (zero) -> orange/red (positive)
DIVERGE = [27, 33, 39, 45, 81, 152, 250, 223, 216, 209, 203, 196]
# white -> yellow -> red, for attention probabilities in [0,1]
HEAT = [236, 238, 242, 248, 223, 222, 220, 214, 208, 202, 196]


def heat_fg(v, vmax):
    """Colour a signed value by magnitude and sign."""
    if vmax <= 1e-9:
        return fg(250)
    t = max(-1.0, min(1.0, v / vmax))
    idx = int((t + 1) / 2 * (len(DIVERGE) - 1) + 0.5)
    return fg(DIVERGE[idx])


def prob_fg(p):
    idx = int(max(0.0, min(1.0, p)) * (len(HEAT) - 1) + 0.5)
    return fg(HEAT[idx])


def clear():
    sys.stdout.write("\033[H\033[J")


def rule(title="", width=None):
    width = width or term_width()
    if not title:
        return DIM + "─" * width + RESET
    head = f"── {title} "
    return DIM + head + "─" * max(0, width - len(head)) + RESET


def term_width():
    return max(80, shutil.get_terminal_size((120, 40)).columns)


# ------------------------------------------------------------------ model ---
torch.manual_seed(SEED)

words = SENTENCE.split()
N = len(words)
VOCAB = sorted(set(words))
IDS = torch.tensor([VOCAB.index(w) for w in words])

# frozen pieces
EMB = torch.randn(len(VOCAB), D_MODEL) * 0.8  # vocab embedding table (frozen)
POS = torch.randn(N, D_MODEL) * 0.30          # positional vectors    (frozen)
X = EMB[IDS] + POS                            # the input matrix      (frozen)
W_O = torch.randn(D_K, D_MODEL) * 0.80       # output projection      (frozen)

# trainable pieces
W_Q = (torch.randn(D_MODEL, D_K) * 0.40).requires_grad_()
W_K = (torch.randn(D_MODEL, D_K) * 0.40).requires_grad_()
W_V = (torch.randn(D_MODEL, D_K) * 0.70).requires_grad_()

CAUSAL = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
TARGETS = IDS[1:]                            # predict the next word

opt = torch.optim.Adam([W_Q, W_K, W_V], lr=LR)


def forward():
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = (Q @ K.T) / math.sqrt(D_K)
    scores = scores.masked_fill(CAUSAL, float("-inf"))
    A = torch.softmax(scores, dim=-1)
    ctx = A @ V                              # (N, D_K)
    logits = ctx @ W_O @ EMB.T               # (N, vocab)
    loss = F.cross_entropy(logits[:-1], TARGETS)
    return Q, K, V, A, logits, loss


def snapshot():
    with torch.no_grad():
        Q, K, V, A, logits, loss = forward()
    return {
        "W_Q": W_Q.detach().clone(), "W_K": W_K.detach().clone(),
        "W_V": W_V.detach().clone(),
        "Q": Q, "K": K, "V": V, "A": A, "logits": logits,
        "loss": float(loss),
    }


def train_step():
    opt.zero_grad()
    *_, loss = forward()
    loss.backward()
    opt.step()
    return float(loss)


# ----------------------------------------------------------------- render ---
CELL_W = 7          # "+0.42" + arrow + space


def render_matrix(cur, prev, row_labels, col_prefix, label_w):
    """Return a list of strings: one header row + one row per matrix row."""
    vmax = float(cur.abs().max())
    d = cur - prev
    dmax = float(d.abs().max())
    lines = []

    head = " " * label_w
    for j in range(cur.shape[1]):
        head += f"{DIM}{col_prefix}{j}{RESET}".rjust(CELL_W + len(DIM) + len(RESET))
    lines.append(head)

    for i in range(cur.shape[0]):
        row = f"{fg(245)}{row_labels[i]:>{label_w - 1}} {RESET}"
        for j in range(cur.shape[1]):
            v = float(cur[i, j])
            dv = float(d[i, j])
            txt = f"{v:+.2f}"
            if abs(dv) < 5e-4:
                arrow = f"{DIM}·{RESET}"
            else:
                strong = dmax > 1e-9 and abs(dv) > 0.45 * dmax
                if dv > 0:
                    arrow = f"{fg(46 if strong else 34)}▲{RESET}"
                else:
                    arrow = f"{fg(203 if strong else 131)}▼{RESET}"
            row += f"{heat_fg(v, vmax)}{txt}{RESET}{arrow} "
        lines.append(row)
    return lines


def block(title, cur, prev, row_labels, col_prefix, label_w):
    body = render_matrix(cur, prev, row_labels, col_prefix, label_w)
    header = f"{BOLD}{title}{RESET}"
    return [header] + body


def visible_len(s):
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
        else:
            out += 1
            i += 1
    return out


def side_by_side(blocks, gap=3):
    height = max(len(b) for b in blocks)
    widths = [max(visible_len(l) for l in b) for b in blocks]
    lines = []
    for r in range(height):
        parts = []
        for b, w in zip(blocks, widths):
            cell = b[r] if r < len(b) else ""
            parts.append(cell + " " * (w - visible_len(cell)))
        lines.append((" " * gap).join(parts))
    return lines


def emit(blocks):
    total = sum(max(visible_len(l) for l in b) for b in blocks) + 3 * (len(blocks) - 1)
    if total <= term_width():
        for l in side_by_side(blocks):
            print(l)
    else:
        for b in blocks:
            for l in b:
                print(l)
            print()


def render_attention(A, prevA):
    w = max(len(x) for x in words)
    d = A - prevA
    print(f"{BOLD}Attention  A = softmax(QKᵀ/√d_k)   "
          f"{DIM}rows = query word, cols = key word, causal mask{RESET}")
    head = " " * (w + 1)
    for j in range(N):
        head += f"{DIM}{words[j][:4]:>5}{RESET}"
    print(head)
    for i in range(N):
        row = f"{fg(245)}{words[i]:>{w}} {RESET}"
        for j in range(N):
            if j > i:
                row += f"{DIM}    ·{RESET}"
            else:
                p = float(A[i, j])
                mark = " "
                if abs(float(d[i, j])) > 0.01:
                    mark = f"{fg(46)}↑{RESET}" if d[i, j] > 0 else f"{fg(203)}↓{RESET}"
                row += f"{prob_fg(p)}{p:>4.2f}{RESET}{mark}"
        # who does this word look at most?
        j = int(A[i].argmax())
        row += f"   {DIM}→ {RESET}{fg(214)}{words[j]}{RESET}"
        print(row)


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(vals, width=48):
    if len(vals) < 2:
        return ""
    v = vals[-width:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-9:
        return SPARK[0] * len(v)
    return "".join(SPARK[int((x - lo) / (hi - lo) * (len(SPARK) - 1))] for x in v)


def render_predictions(logits):
    print(f"{BOLD}Next-word prediction{RESET} {DIM}(what the head currently outputs){RESET}")
    line_a, line_b = "", ""
    for i in range(N - 1):
        p = torch.softmax(logits[i], dim=-1)
        j = int(p.argmax())
        ok = j == int(IDS[i + 1])
        col = fg(46) if ok else fg(203)
        pred = f"{VOCAB[j][:9]}"
        line_a += f"{DIM}{words[i][:9]:>10}{RESET}"
        line_b += f"{col}{pred:>10}{RESET}"
    print(f"{DIM}   given{RESET}" + line_a)
    print(f"{DIM} predict{RESET}" + line_b)
    print(f"{DIM}  truth {RESET}" + "".join(f"{DIM}{w[:9]:>10}{RESET}" for w in words[1:]))


# ------------------------------------------------------------------ frames ---
def frame_header(step, cur, prev, base, mode, losses):
    clear()
    dl = cur["loss"] - prev["loss"]
    dcol = fg(46) if dl < 0 else fg(203)
    print(rule(f"STEP {step}  ·  d_model={D_MODEL}  d_k={D_K}  Adam lr={LR}"))
    print(f'  {DIM}sentence:{RESET} "{fg(214)}{SENTENCE}{RESET}"')
    correct = int((cur["logits"][:-1].argmax(-1) == TARGETS).sum())
    print(f"  {DIM}loss{RESET} {BOLD}{cur['loss']:.4f}{RESET}  "
          f"{dcol}{dl:+.4f}{RESET}   {DIM}{sparkline(losses)}{RESET}   "
          f"{DIM}next-word correct{RESET} {BOLD}{correct}/{N - 1}{RESET}")
    drift = " ".join(
        f"{name} ‖Δ‖={float((cur[name] - base[name]).norm()):.3f}"
        for name in ("W_Q", "W_K", "W_V"))
    print(f"  {DIM}drift from init:  {drift}{RESET}")
    print(f"  {DIM}▲▼ compares against {'the previous step' if mode == 'step' else 'step 0 (init)'}"
          f"  ·  press [d] to switch{RESET}")
    print()


def draw(step, cur, prev, base, mode, losses):
    ref = prev if mode == "step" else base
    frame_header(step, cur, prev, base, mode, losses)

    print(rule(f"learned weight matrices   ({D_MODEL} x {D_K})"))
    dims = [f"d{i}" for i in range(D_MODEL)]
    emit([
        block("W_Q", cur["W_Q"], ref["W_Q"], dims, "q", 5),
        block("W_K", cur["W_K"], ref["W_K"], dims, "k", 5),
        block("W_V", cur["W_V"], ref["W_V"], dims, "v", 5),
    ])
    print()

    print(rule("projected matrices   Q = X·W_Q     K = X·W_K     V = X·W_V   (9 x %d)" % D_K + ""))
    lw = max(len(w) for w in words) + 1
    emit([
        block("Q  (queries)", cur["Q"], ref["Q"], words, "q", lw),
        block("K  (keys)", cur["K"], ref["K"], words, "k", lw),
        block("V  (values)", cur["V"], ref["V"], words, "v", lw),
    ])
    print()

    print(rule("what Q and K produce"))
    render_attention(cur["A"], ref["A"])
    print()
    render_predictions(cur["logits"])
    print()
    print(rule())


# -------------------------------------------------------------- intro pages --
def page_intro():
    clear()
    print(rule("A TINY ATTENTION HEAD, LEARNING"))
    print(f"""
  Sentence ({N} words):   "{fg(214)}{SENTENCE}{RESET}"

  The vocabulary is just the words in this sentence. The model is a single
  self-attention head:

      {fg(45)}Q = X · W_Q      K = X · W_K      V = X · W_V{RESET}
      A = softmax( Q Kᵀ / √d_k ),  causally masked
      out = A · V · W_O · Embᵀ   ->  next-word logits

  {BOLD}Frozen{RESET}   : the embeddings X, the positional vectors, W_O, the unembedding
  {BOLD}Trained{RESET}  : {fg(214)}W_Q, W_K, W_V{RESET} only  — so every number that moves,
             moves because of them.

  d_model = {D_MODEL}, d_k = {D_K}, optimiser = Adam(lr={LR}),
  loss = cross-entropy on causal next-word prediction.

  {DIM}Reading the display{RESET}
    {heat_fg(-1, 1)}blue{RESET} = negative   {heat_fg(1, 1)}orange/red{RESET} = positive   (shade = magnitude)
    {fg(46)}▲{RESET} value went up   {fg(203)}▼{RESET} value went down   {DIM}·{RESET} unchanged

  {DIM}Controls{RESET}
    [Enter] one training step     [number]+Enter  run that many steps
    [d] toggle Δ-vs-previous / Δ-vs-init          [q] quit
""")
    print(rule())
    input(f"  {DIM}Enter to see the frozen input matrix X …{RESET}")


def page_inputs():
    clear()
    print(rule("STEP 0a  ·  the frozen input  X = Embedding + Position   (9 x %d)" % D_MODEL + ""))
    print(f"  {DIM}This never changes during training. It is the raw material that"
          f"\n  W_Q, W_K and W_V will learn to reshape.{RESET}\n")
    lw = max(len(w) for w in words) + 1
    zero = torch.zeros_like(X)
    emit([block("X", X, X + zero, words, "d", lw)])
    print()
    print(rule())
    input(f"  {DIM}Enter to see the randomly initialised W_Q, W_K, W_V …{RESET}")


def page_random_weights(base):
    clear()
    print(rule("STEP 0b  ·  random initialisation of W_Q, W_K, W_V   (%d x %d)" % (D_MODEL, D_K) + ""))
    print(f"  {DIM}Drawn from a normal distribution. No structure, no meaning yet.{RESET}\n")
    dims = [f"d{i}" for i in range(D_MODEL)]
    emit([
        block("W_Q", base["W_Q"], base["W_Q"], dims, "q", 5),
        block("W_K", base["W_K"], base["W_K"], dims, "k", 5),
        block("W_V", base["W_V"], base["W_V"], dims, "v", 5),
    ])
    print()
    print(rule("the Q, K, V they produce"))
    lw = max(len(w) for w in words) + 1
    emit([
        block("Q  (queries)", base["Q"], base["Q"], words, "q", lw),
        block("K  (keys)", base["K"], base["K"], words, "k", lw),
        block("V  (values)", base["V"], base["V"], words, "v", lw),
    ])
    print()
    render_attention(base["A"], base["A"])
    correct = int((base["logits"][:-1].argmax(-1) == TARGETS).sum())
    print(f"\n  {DIM}Attention is smeared across the allowed positions and the arrows on"
          f"\n  the right point nowhere meaningful — the head has no idea what to look"
          f"\n  at. loss = {base['loss']:.4f}, {correct}/{N - 1} next words correct.{RESET}")
    print(rule())
    input(f"  {DIM}Enter to start training …{RESET}")


# -------------------------------------------------------------------- main ---
def main():
    base = snapshot()
    losses = [base["loss"]]
    mode = "step"

    page_intro()
    page_inputs()
    page_random_weights(base)

    prev = base
    step = 0
    pending = 0

    while step < MAX_STEPS:
        if pending > 0:
            pending -= 1
        else:
            try:
                cmd = input(f"  {BOLD}[Enter]{RESET} step  "
                            f"{DIM}| number = run N | d = delta mode | q = quit >{RESET} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if cmd == "q":
                break
            if cmd == "d":
                mode = "init" if mode == "step" else "step"
                draw(step, prev if step else base, prev, base, mode, losses)
                continue
            if cmd.isdigit():
                pending = max(0, int(cmd) - 1)

        loss_before = train_step()
        step += 1
        cur = snapshot()
        losses.append(cur["loss"])
        draw(step, cur, prev, base, mode, losses)
        prev = cur

    clear()
    print(rule("DONE"))
    print(f"\n  steps run     : {step}")
    print(f"  loss          : {losses[0]:.4f}  →  {losses[-1]:.4f}")
    print(f"  {DIM}{sparkline(losses, width=70)}{RESET}\n")
    for name in ("W_Q", "W_K", "W_V"):
        print(f"  {name}: ‖Δ from init‖ = {float((prev[name] - base[name]).norm()):.4f}")
    print()
    with torch.no_grad():
        *_, A, logits, _ = forward()
    print(f"  next-word correct: {int((logits[:-1].argmax(-1) == TARGETS).sum())}/{N - 1}\n")
    render_attention(A, base["A"])
    print()
    render_predictions(logits)
    print()
    print(rule())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(RESET)
