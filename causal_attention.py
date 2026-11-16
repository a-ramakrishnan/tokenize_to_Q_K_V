# Ramakrishnan A
# File 9-A in the series on Building LLMs from scratch
# Causal (Masked) Self-Attention: hiding future tokens so each word can only
# attend to itself and earlier words in the sentence (no Dropout yet).
# Lots of pranams to Sebastian Raschka

import torch.nn as nn
import torch
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Reuse the word2vec-backed embedding generator (with local caching) from
# File 6, rather than duplicating it here. That file's execution code is
# wrapped in `if __name__ == "__main__":`, so importing it only pulls in
# the function/constant definitions -- no side effects.
from attention_without_trainable_weights import generate_semantic_embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Causal (masked) self-attention with trainable Q/K/V weights, "
                    "using word2vec embeddings for the input sentence."
    )
    parser.add_argument("sentence", type=str, help="The sentence to tokenize and embed")
    parser.add_argument("--pca-init", action="store_true",
                        help="Initialize WQuery/WKey/WValue from PCA of the input "
                             "embeddings instead of random weights -- no training, "
                             "but tends to produce more semantically meaningful "
                             "attention than a random projection at low d_out.")
    return parser.parse_args()


def plot_attention_heatmap(input_words, attention_weights, title, figsize=(9, 8), save_path=None, show=True):
    """
    Visualize an attention weights matrix as a heatmap, with words
    labeled on both axes and each cell annotated with its value.
    """
    weights_np = attention_weights.detach().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(weights_np, cmap='viridis', vmin=0, vmax=weights_np.max())

    ax.set_xticks(range(len(input_words)))
    ax.set_yticks(range(len(input_words)))
    ax.set_xticklabels(input_words, rotation=45, ha='right')
    ax.set_yticklabels(input_words)
    ax.set_xlabel('Key (attended to)')
    ax.set_ylabel('Query (attending from)')
    ax.set_title(title)

    for i in range(len(input_words)):
        for j in range(len(input_words)):
            value = weights_np[i, j]
            text_color = 'white' if value < weights_np.max() * 0.6 else 'black'
            ax.text(j, i, f"{value:.2f}", ha='center', va='center', color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_causal_masking_pipeline(input_words, raw_scores, mask_bool, masked_scores,
                                 unmasked_weights, causal_weights,
                                 figsize=(18, 5.5), save_path=None, show=True):
    """
    Visualize how the causal mask turns full self-attention into causal
    (left-to-right) self-attention across 4 panels:
      1. Scaled Attention Scores (queries @ keys.T / sqrt(d_k))
      2. Upper-Triangular Causal Mask (1 = future token to hide, 0 = allowed)
      3. Masked Scores (-inf above the diagonal, untouched on/below)
      4. Final Causal Attention Weights after softmax (strictly lower-triangular)
    Plus a 2-panel comparison of Unmasked vs. Causal Attention Weights.
    """
    scores_np = raw_scores.detach().numpy()
    mask_np = mask_bool.detach().numpy().astype(float)
    masked_np = masked_scores.detach().numpy()
    causal_np = causal_weights.detach().numpy()
    unmasked_np = unmasked_weights.detach().numpy()
    n = len(input_words)

    # --- Figure 1: 4-step causal masking pipeline ---
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Panel 1: Scaled raw scores
    im0 = axes[0].imshow(scores_np, cmap='coolwarm', aspect='auto')
    axes[0].set_title("1. Scaled Scores\n(queries @ keys.T / √d_k)", fontsize=10)

    # Panel 2: Upper-triangular mask
    im1 = axes[1].imshow(mask_np, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title("2. Causal Mask (triu, diag=1)\n(Dark = Future Blocked)", fontsize=10)

    # Panel 3: Masked scores (-inf in upper triangle)
    finite_vals = scores_np[~mask_bool.detach().numpy()]
    vmin_s, vmax_s = finite_vals.min(), finite_vals.max()
    masked_display = np.where(np.isinf(masked_np), np.nan, masked_np)
    cmap_masked = plt.cm.coolwarm.copy()
    cmap_masked.set_bad(color='#2b2b2b')
    im2 = axes[2].imshow(masked_display, cmap=cmap_masked, vmin=vmin_s, vmax=vmax_s, aspect='auto')
    axes[2].set_title("3. Masked Scores\n(Upper Triangle = -∞)", fontsize=10)

    # Panel 4: Causal attention weights after softmax
    im3 = axes[3].imshow(causal_np, cmap='viridis', vmin=0, vmax=causal_np.max(), aspect='auto')
    axes[3].set_title("4. Causal Weights (after softmax)\n(Future = 0.00, Rows sum to 1)", fontsize=10)

    for idx, ax in enumerate(axes):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(input_words, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(input_words if idx == 0 else [], fontsize=8)
        if idx == 0:
            ax.set_ylabel("Query (attending from)")
        ax.set_xlabel("Key (attended to)")

        # Annotate cells if sentence is reasonably short
        if n <= 12:
            for i in range(n):
                for j in range(n):
                    if idx == 0:
                        val = scores_np[i, j]
                        txt = f"{val:.2f}"
                        col = 'white' if abs(val) > np.abs(scores_np).max() * 0.6 else 'black'
                    elif idx == 1:
                        blocked = mask_np[i, j] > 0.5
                        txt = "MASK" if blocked else "OK"
                        col = 'white' if blocked else 'black'
                    elif idx == 2:
                        if np.isinf(masked_np[i, j]):
                            txt = "-∞"
                            col = '#ff6b6b'
                        else:
                            val = masked_np[i, j]
                            txt = f"{val:.2f}"
                            col = 'white' if abs(val) > max(abs(vmin_s), abs(vmax_s)) * 0.6 else 'black'
                    else:
                        val = causal_np[i, j]
                        txt = f"{val:.2f}"
                        col = 'white' if val < causal_np.max() * 0.6 else 'black'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=6.5, color=col)

    plt.suptitle("How Causal Masking Hides Future Tokens Before Softmax", fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # --- Figure 2: Side-by-side Unmasked vs. Causal Attention Weights ---
    fig2, (ax_u, ax_c) = plt.subplots(1, 2, figsize=(14, 6))
    vmax_both = max(unmasked_np.max(), causal_np.max())

    for ax, mat, panel_title in [
        (ax_u, unmasked_np, "Unmasked Self-Attention (File 7)\nEvery word sees past AND future words"),
        (ax_c, causal_np, "Causal Self-Attention (File 9-A)\nEach word sees ONLY itself and earlier words"),
    ]:
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=vmax_both)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(input_words, rotation=45, ha='right')
        ax.set_yticklabels(input_words)
        ax.set_xlabel('Key (attended to)')
        ax.set_ylabel('Query (attending from)')
        ax.set_title(panel_title, fontsize=10.5)
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                col = 'white' if val < vmax_both * 0.6 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=col, fontsize=7.5)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig2)


def compute_causal_attention_entropy(input_words, attention_weights):
    """
    Compute and print the Shannon entropy of each word's causal attention
    distribution.
    In causal attention, word i (0-indexed) can only see (i + 1) words
    (positions 0..i), so its maximum possible entropy is log(i + 1) rather
    than log(N). Word 0 only sees itself, so its entropy is always 0.0000.
    """
    weights_np = attention_weights.detach().numpy()
    n = weights_np.shape[1]

    print("\nCausal Attention Entropy per word (word i sees i+1 words, so max entropy = log(i+1)):")
    print(f"{'Pos':<5}{'Word':<12}{'Visible':<9}{'Entropy':<12}{'Max log(i+1)':<14}{'% of Causal Max':<15}")
    print("-" * 67)

    entropies = []
    pcts = []
    for i, (word, row) in enumerate(zip(input_words, weights_np)):
        visible_count = i + 1
        local_max_entropy = np.log(visible_count)
        # Only sum over non-zero/visible positions (0..i)
        visible_probs = row[:visible_count]
        entropy = -np.sum(visible_probs * np.log(visible_probs + 1e-12))
        entropy = max(0.0, float(entropy))
        entropies.append(entropy)
        if local_max_entropy > 0:
            pct = 100.0 * entropy / local_max_entropy
            pcts.append(pct)
            pct_str = f"{pct:.1f}%"
        else:
            pct_str = "N/A (1 word)"
        print(f"{i:<5}{word:<12}{visible_count:<9}{entropy:<12.4f}{local_max_entropy:<14.4f}{pct_str:<15}")

    avg_entropy = np.mean(entropies)
    avg_pct = np.mean(pcts) if pcts else 0.0
    print(f"\nAverage causal entropy: {avg_entropy:.4f} (average {avg_pct:.1f}% of each position's causal max)")
    return entropies


def print_top_k_attended(input_words, attention_weights, k=3):
    """
    Print, in plain English, which visible (past/present) words each word attends to most.
    """
    weights_np = attention_weights.detach().numpy()
    print(f"\nTop-{k} attended visible words per word (causal -- future words have 0.00 weight):")
    for i, word in enumerate(input_words):
        row = weights_np[i, :i + 1]  # only positions 0..i are visible
        top_indices = row.argsort()[::-1][:min(k, i + 1)]
        parts = [f"{input_words[idx]} ({row[idx]:.2f})" for idx in top_indices]
        print(f'  [{i}] "{word}" (sees {i + 1} word{"s" if i > 0 else ""}) attends most to: ' + ", ".join(parts))


def plot_matrix_multiplication(A, B, C, A_label, B_label, C_label,
                                A_row_labels=None, C_row_labels=None,
                                cmap='coolwarm', figsize=(15, 5),
                                save_path=None, show=True):
    """
    Visualize A @ B = C as three heatmaps side by side, connected by
    x and = symbols -- makes the shapes and the multiplication itself
    visible, not just the resulting numbers.
    """
    A = A.detach().numpy() if hasattr(A, "detach") else np.array(A)
    B = B.detach().numpy() if hasattr(B, "detach") else np.array(B)
    C = C.detach().numpy() if hasattr(C, "detach") else np.array(C)

    widths = [A.shape[1], 0.6, B.shape[1], 0.6, C.shape[1]]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 5, width_ratios=widths, figure=fig)

    def draw_matrix(ax, M, title, row_labels=None):
        ax.imshow(M, cmap=cmap, aspect='auto')
        ax.set_title(f"{title}\nshape {M.shape}", fontsize=10)
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
        if row_labels is not None:
            ax.set_yticklabels(row_labels, fontsize=7)
        if M.shape[0] * M.shape[1] <= 150:
            vmax = np.abs(M).max()
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    color = 'white' if abs(M[i, j]) > vmax * 0.6 else 'black'
                    ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center',
                            fontsize=6, color=color)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_matrix(ax_a, A, A_label, A_row_labels)

    ax_x = fig.add_subplot(gs[0, 1]); ax_x.axis('off')
    ax_x.text(0.5, 0.5, '×', ha='center', va='center', fontsize=28)

    ax_b = fig.add_subplot(gs[0, 2])
    draw_matrix(ax_b, B, B_label)

    ax_eq = fig.add_subplot(gs[0, 3]); ax_eq.axis('off')
    ax_eq.text(0.5, 0.5, '=', ha='center', va='center', fontsize=28)

    ax_c = fig.add_subplot(gs[0, 4])
    draw_matrix(ax_c, C, C_label, C_row_labels)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


class CausalSelfAttention(nn.Module):
    """
    Causal (Masked) Self-Attention (without Dropout).

    Why we need a causal mask in GPT-style language models:
    When predicting the next word after position i, the model must ONLY use
    information from positions 0..i. If position i could attend to position
    j > i, it would be peeking at the ground-truth answer it is trying to
    predict!

    Two ways to mask future tokens are demonstrated inside forward():
      - Approach A (Naive): softmax first -> zero out upper triangle with
        torch.tril -> re-normalize rows so they sum to 1 again.
      - Approach B (Standard / Efficient): fill upper triangle of scores
        with -inf BEFORE softmax using a registered buffer (`register_buffer`).
        Since exp(-inf) = 0, softmax produces 0s for future tokens and rows
        that already sum to 1.0 in a single step, with zero information leakage
        from future logits into the denominator.
    """
    def __init__(self, d_in, d_out, context_length, qkv_bias=False, visualize=False, word_labels=None):
        super().__init__()
        self.d_out = d_out
        self.context_length = context_length
        self.visualize = visualize
        self.word_labels = word_labels

        self.WQuery = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WKey = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WValue = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        # Register the upper-triangular mask (1s above the main diagonal, 0s on
        # and below it) as a non-trainable buffer. Using register_buffer means:
        #   1. PyTorch does NOT treat `mask` as a trainable weight (no gradients).
        #   2. It automatically moves to GPU/MPS with `.to(device)` and is saved
        #      in `state_dict()`.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # Populated by forward() so they can be inspected/visualized afterward
        self.last_queries = None
        self.last_keys = None
        self.last_values = None
        self.last_attention_scores = None
        self.last_scaled_scores = None
        self.last_masked_scores = None
        self.last_unmasked_weights = None
        self.last_attention_weights = None

    def init_qkv_from_pca(self, x):
        """
        Initialize WQuery, WKey, WValue using the top principal components
        of the actual input embeddings (SVD, no gradient training).
        """
        with torch.no_grad():
            x_np = x.detach().numpy()
            centered = x_np - x_np.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            d_out = self.WQuery.weight.shape[0]
            pca_components = Vt[:d_out]
            pca_tensor = torch.tensor(pca_components, dtype=torch.float32)
            self.WQuery.weight.copy_(pca_tensor)
            self.WKey.weight.copy_(pca_tensor)
            self.WValue.weight.copy_(pca_tensor)
        print("Initialized WQuery/WKey/WValue from PCA of the input embeddings "
              "(no training, just SVD).")

    def forward(self, x):
        # Support both 2D (num_tokens, d_in) -- like File 7 -- and 3D batched
        # (batch, num_tokens, d_in) inputs.
        is_batched = (x.dim() == 3)
        if not is_batched:
            num_tokens, _ = x.shape
        else:
            _, num_tokens, _ = x.shape

        print("=" * 70)
        print("STEP 1: Project the input into Query, Key, and Value spaces")
        print("=" * 70)
        print(f"Input x, shape {tuple(x.shape)}:\n{x}\n")

        queries = self.WQuery(x)
        keys = self.WKey(x)
        values = self.WValue(x)

        print(f"queries = WQuery(x), shape {tuple(queries.shape)}:\n{queries}\n")
        print(f"keys    = WKey(x),   shape {tuple(keys.shape)}:\n{keys}\n")
        print(f"values  = WValue(x), shape {tuple(values.shape)}:\n{values}\n")

        print("=" * 70)
        print("STEP 2: Raw Attention Scores = queries @ keys.T")
        print("=" * 70)
        # Transpose the last two dimensions so it works for both 2D and 3D tensors
        attentionScores = queries @ keys.transpose(-2, -1)
        d_k = keys.shape[-1]
        scaledScores = attentionScores / (d_k ** 0.5)
        print(f"attentionScores, shape {tuple(attentionScores.shape)}:\n{attentionScores}\n")
        print(f"scaledScores = attentionScores / sqrt({d_k}), shape {tuple(scaledScores.shape)}:\n{scaledScores}\n")

        # Keep unmasked weights around purely to compare against causal weights
        unmaskedWeights = torch.softmax(scaledScores, dim=-1)

        print("=" * 70)
        print("STEP 3A (Naive Walkthrough): Softmax -> Zero Upper Triangle -> Renormalize")
        print("=" * 70)
        tril_mask = torch.tril(torch.ones(num_tokens, num_tokens, device=x.device))
        print(f"Lower-triangular mask (torch.tril):\n{tril_mask}\n")
        masked_simple = unmaskedWeights * tril_mask
        print(f"After multiplying softmax weights by tril_mask (rows no longer sum to 1!):\n{masked_simple}")
        row_sums = masked_simple.sum(dim=-1, keepdim=True)
        print(f"Row sums before re-normalization: {row_sums.squeeze(-1)}\n")
        masked_simple_norm = masked_simple / row_sums
        print(f"After dividing each row by its sum (rows sum to 1 again):\n{masked_simple_norm}\n")

        print("=" * 70)
        print("STEP 3B (Standard Transformer Way): Mask with -inf BEFORE Softmax")
        print("=" * 70)
        print("Why this is better than Step 3A: by setting future positions to -inf")
        print("before softmax, exp(-inf) = 0, so future tokens never influence the")
        print("softmax denominator at all, and we avoid a separate division pass.\n")

        # Slice the pre-registered upper-triangular mask to the actual sequence length
        causal_mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        print(f"Upper-triangular boolean mask (True = future position to mask out):\n{causal_mask_bool}\n")

        maskedScores = scaledScores.masked_fill(causal_mask_bool, -torch.inf)
        print(f"maskedScores (after masked_fill with -inf):\n{maskedScores}\n")

        attentionWeights = torch.softmax(maskedScores, dim=-1)
        print(f"Causal attentionWeights = softmax(maskedScores), shape {tuple(attentionWeights.shape)}:")
        print(f"{attentionWeights}\n")
        print(f"Row sums (all 1.0 in a single pass): {attentionWeights.sum(dim=-1)}")
        print("Max difference between Naive (Step 3A) and -inf Softmax (Step 3B): "
              f"{torch.max(torch.abs(masked_simple_norm - attentionWeights)).item():.2e}\n")

        print("=" * 70)
        print("STEP 4: Causal Context Vector = attentionWeights @ values")
        print("=" * 70)
        context_vector = attentionWeights @ values
        print(f"context_vector, shape {tuple(context_vector.shape)}:\n{context_vector}\n")
        print("Notice Row 0 of context_vector equals Row 0 of values (1.00 * values[0]),")
        print("because the first word can only attend to itself!\n")

        if self.visualize and not is_batched:
            plot_causal_masking_pipeline(
                self.word_labels, scaledScores, causal_mask_bool,
                maskedScores, unmaskedWeights, attentionWeights
            )
            plot_matrix_multiplication(
                attentionWeights, values, context_vector,
                A_label="Causal Attention Weights", B_label="Values", C_label="Causal Context Vector",
                A_row_labels=self.word_labels, C_row_labels=self.word_labels,
            )

        # Stash intermediate tensors for inspection/visualization after forward()
        self.last_queries = queries
        self.last_keys = keys
        self.last_values = values
        self.last_attention_scores = attentionScores
        self.last_scaled_scores = scaledScores
        self.last_masked_scores = maskedScores
        self.last_unmasked_weights = unmaskedWeights
        self.last_attention_weights = attentionWeights

        return context_vector


if __name__ == "__main__":
    args = parse_args()
    input_words = re.findall(r"\b[\w']+\b", args.sentence)

    print("Sentence Used:", args.sentence)
    print("Words:", input_words)

    # Use 6D word2vec-reduced embeddings (matching File 7-D)
    inputs = generate_semantic_embeddings(input_words, output_dim=6)
    print("Embeddings (from word2vec, reduced to 6D):\n", inputs)

    context_length = inputs.shape[0]
    d_in = inputs.shape[1]
    d_out = inputs.shape[1] - 1

    torch.manual_seed(15033019)
    causal_attention = CausalSelfAttention(
        d_in, d_out, context_length=context_length,
        visualize=True, word_labels=input_words
    )
    if args.pca_init:
        print("\nWQuery.weight BEFORE pca-init:\n", causal_attention.WQuery.weight.data)
        causal_attention.init_qkv_from_pca(inputs)
        print("WQuery.weight AFTER pca-init:\n", causal_attention.WQuery.weight.data)

    context_vector = causal_attention(inputs)

    print("=" * 70)
    print("FINAL: Causal Context Vector")
    print("=" * 70)
    print(context_vector)

    plot_attention_heatmap(
        input_words, causal_attention.last_attention_weights,
        title="Causal Attention Weights Heatmap (Future Tokens Masked)"
    )
    compute_causal_attention_entropy(input_words, causal_attention.last_attention_weights)
    print_top_k_attended(input_words, causal_attention.last_attention_weights, k=3)
