# Ramakrishnan A
# File 9-B in the series on Building LLMs from scratch
# Causal (Masked) Self-Attention with Attention Weight Dropout: randomly
# dropping attention weights during training and scaling survivors by 1/(1-p).
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
        description="Causal (masked) self-attention with trainable Q/K/V weights "
                    "and attention-weight dropout, using word2vec embeddings."
    )
    parser.add_argument("sentence", type=str, help="The sentence to tokenize and embed")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Attention dropout probability p in [0, 1). Default is 0.5 "
                             "(50%%, matching Sebastian Raschka's walkthrough so the "
                             "1/(1-p) = 2x rescaling is easy to spot; 0.1-0.2 is typical "
                             "in full GPT training).")
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
    im = ax.imshow(weights_np, cmap='viridis', vmin=0, vmax=max(weights_np.max(), 1e-6))

    ax.set_xticks(range(len(input_words)))
    ax.set_yticks(range(len(input_words)))
    ax.set_xticklabels(input_words, rotation=45, ha='right')
    ax.set_yticklabels(input_words)
    ax.set_xlabel('Key (attended to)')
    ax.set_ylabel('Query (attending from)')
    ax.set_title(title)

    vmax = max(weights_np.max(), 1e-6)
    for i in range(len(input_words)):
        for j in range(len(input_words)):
            value = weights_np[i, j]
            text_color = 'white' if value < vmax * 0.6 else 'black'
            ax.text(j, i, f"{value:.2f}", ha='center', va='center', color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_dropout_pipeline(input_words, pre_dropout_weights, post_dropout_weights,
                          dropout_p, figsize=(18, 5.5), save_path=None, show=True):
    """
    Visualize what nn.Dropout(p) does to the causal attention weight matrix
    across 4 panels:
      1. Pre-Dropout Causal Weights (after softmax, rows sum to 1.0)
      2. Active Dropout Mask on Visible Cells (Kept vs. Dropped to 0)
      3. Post-Dropout Causal Weights (survivors scaled up by 1 / (1 - p))
      4. Weight Shift (Post - Pre: blue = dropped, red = boosted survivor)
    """
    pre_np = pre_dropout_weights.detach().numpy()
    post_np = post_dropout_weights.detach().numpy()
    diff_np = post_np - pre_np
    n = len(input_words)
    scale_factor = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 0.0

    # Classify each cell:
    #   0 = future token (already masked by causal mask)
    #   1 = visible token dropped by dropout (pre > 0, post == 0)
    #   2 = visible token kept & scaled by 1/(1-p)
    cell_state = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            if pre_np[i, j] > 0 and post_np[i, j] == 0:
                cell_state[i, j] = 1.0  # dropped
            else:
                cell_state[i, j] = 2.0  # kept

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    vmax_w = max(pre_np.max(), post_np.max(), 1e-6)

    # Panel 1: Pre-dropout causal weights
    axes[0].imshow(pre_np, cmap='viridis', vmin=0, vmax=vmax_w, aspect='auto')
    axes[0].set_title("1. Pre-Dropout Causal Weights\n(Rows sum to 1.0)", fontsize=10)

    # Panel 2: Which visible weights were kept vs dropped
    from matplotlib.colors import ListedColormap
    mask_cmap = ListedColormap(['#2b2b2b', '#e03131', '#2f9e44'])
    axes[1].imshow(cell_state, cmap=mask_cmap, vmin=0, vmax=2, aspect='auto')
    axes[1].set_title(
        f"2. Dropout Decisions (p={dropout_p:.2f})\n(Green=Kept, Red=Dropped, Dark=Future)",
        fontsize=10
    )

    # Panel 3: Post-dropout causal weights
    axes[2].imshow(post_np, cmap='viridis', vmin=0, vmax=vmax_w, aspect='auto')
    axes[2].set_title(
        f"3. Post-Dropout Weights\n(Survivors scaled ×{scale_factor:.2f})",
        fontsize=10
    )

    # Panel 4: Difference (Post - Pre)
    max_abs_diff = max(np.abs(diff_np).max(), 1e-6)
    axes[3].imshow(diff_np, cmap='coolwarm', vmin=-max_abs_diff, vmax=max_abs_diff, aspect='auto')
    axes[3].set_title("4. Change (Post − Pre)\n(Blue = Zeroed Out, Red = Boosted)", fontsize=10)

    for idx, ax in enumerate(axes):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(input_words, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(input_words if idx == 0 else [], fontsize=8)
        if idx == 0:
            ax.set_ylabel("Query (attending from)")
        ax.set_xlabel("Key (attended to)")

        if n <= 12:
            for i in range(n):
                for j in range(n):
                    if idx == 0:
                        val = pre_np[i, j]
                        txt = f"{val:.2f}"
                        col = 'white' if val < vmax_w * 0.6 else 'black'
                    elif idx == 1:
                        if j > i:
                            txt = "—"
                            col = '#aaaaaa'
                        elif cell_state[i, j] == 1.0:
                            txt = "DROP"
                            col = 'white'
                        else:
                            txt = "KEEP"
                            col = 'white'
                    elif idx == 2:
                        val = post_np[i, j]
                        txt = f"{val:.2f}"
                        col = 'white' if val < vmax_w * 0.6 else 'black'
                    else:
                        val = diff_np[i, j]
                        txt = f"{val:+.2f}" if j <= i else "0.00"
                        col = 'white' if abs(val) > max_abs_diff * 0.5 else 'black'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=6.5, color=col)

    plt.suptitle(
        f"How Attention Dropout (p={dropout_p}) Drops Links and Rescales Survivors by 1/(1-p) = {scale_factor:.2f}x",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_train_vs_eval_comparison(input_words, train_weights, eval_weights, dropout_p,
                                  figsize=(14, 6), save_path=None, show=True):
    """
    Compare the attention weights produced in model.train() mode (dropout active)
    against model.eval() mode (dropout automatically disabled for inference).
    """
    train_np = train_weights.detach().numpy()
    eval_np = eval_weights.detach().numpy()
    n = len(input_words)
    vmax_both = max(train_np.max(), eval_np.max(), 1e-6)

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=figsize)
    for ax, mat, panel_title in [
        (ax_t, train_np,
         f"Training Mode: model.train() (dropout={dropout_p})\n"
         f"Random subset dropped to 0, survivors scaled by 1/(1-{dropout_p:.2f})"),
        (ax_e, eval_np,
         "Inference Mode: model.eval() (dropout automatically off)\n"
         "All causal weights intact, every row sums to 1.00"),
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


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
            vmax = max(np.abs(M).max(), 1e-6)
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


def print_dropout_summary(input_words, pre_weights, post_weights, dropout_p):
    """
    Print a word-by-word breakdown of how many visible attention links
    each word had before dropout, how many survived, and which words it
    attends to after dropout.
    """
    pre_np = pre_weights.detach().numpy()
    post_np = post_weights.detach().numpy()
    scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 0.0

    print(f"\nDropout Summary per word (p = {dropout_p:.2f}, survivor scale = 1/(1-p) = {scale:.2f}x):")
    print(f"{'Pos':<5}{'Word':<12}{'Visible':<9}{'Kept':<7}{'Dropped':<9}{'Row Sum After':<15}{'Surviving Attended Words'}")
    print("-" * 88)

    total_visible = 0
    total_dropped = 0
    for i, word in enumerate(input_words):
        visible = i + 1
        row_post = post_np[i, :visible]
        kept = int(np.sum(row_post > 0))
        dropped = visible - kept
        total_visible += visible
        total_dropped += dropped
        row_sum = float(row_post.sum())

        if kept == 0:
            attended_str = "(all dropped -> zero context vector)"
        else:
            top_idx = row_post.argsort()[::-1]
            parts = [f"{input_words[idx]} ({row_post[idx]:.2f})" for idx in top_idx if row_post[idx] > 0][:3]
            attended_str = ", ".join(parts)

        print(f"{i:<5}{word:<12}{visible:<9}{kept:<7}{dropped:<9}{row_sum:<15.2f}{attended_str}")

    actual_drop_rate = 100.0 * total_dropped / total_visible if total_visible > 0 else 0.0
    print("-" * 88)
    print(f"Total visible causal links: {total_visible} | Dropped: {total_dropped} "
          f"({actual_drop_rate:.1f}% actual vs {dropout_p * 100:.1f}% target p)")


class CausalAttention(nn.Module):
    """
    Complete Causal Self-Attention class with Dropout (matching Sebastian
    Raschka's `CausalAttention` in Chapter 3.5.3).

    Supports both:
      - 2D unbatched input: (num_tokens, d_in)
      - 3D batched input  : (batch_size, num_tokens, d_in)

    What Dropout adds on top of File 9-A (`causal_attention.py`):
      1. After computing the causal attention weights via `softmax`, we pass
         them through `nn.Dropout(dropout)`.
      2. During training (`model.train()`), each attention weight has a
         probability `p` of being zeroed out, and all surviving weights are
         automatically rescaled by `1 / (1 - p)` (Inverted Dropout) so the
         expected sum of weights remains 1.0.
      3. During evaluation/inference (`model.eval()`), `nn.Dropout` becomes a
         no-op automatically.
    """
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False,
                 visualize=False, word_labels=None):
        super().__init__()
        self.d_out = d_out
        self.context_length = context_length
        self.dropout_p = dropout
        self.visualize = visualize
        self.word_labels = word_labels

        self.WQuery = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WKey = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WValue = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        # Attention dropout layer applied directly to the softmax weights
        self.dropout = torch.nn.Dropout(dropout)

        # Non-trainable upper-triangular causal mask buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # Populated by forward() so they can be inspected/visualized afterward
        self.last_queries = None
        self.last_keys = None
        self.last_values = None
        self.last_attention_scores = None
        self.last_masked_scores = None
        self.last_pre_dropout_weights = None
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
        is_batched = (x.dim() == 3)
        if not is_batched:
            num_tokens, _ = x.shape
        else:
            _, num_tokens, _ = x.shape

        print("=" * 70)
        print("STEP 1: Project the input into Query, Key, and Value spaces")
        print("=" * 70)
        queries = self.WQuery(x)
        keys = self.WKey(x)
        values = self.WValue(x)
        print(f"Input x shape: {tuple(x.shape)} -> Q, K, V shape: {tuple(queries.shape)}\n")

        print("=" * 70)
        print("STEP 2 & 3: Scaled Scores + Causal Mask (-inf) + Softmax")
        print("=" * 70)
        attentionScores = queries @ keys.transpose(-2, -1)
        d_k = keys.shape[-1]
        scaledScores = attentionScores / (d_k ** 0.5)

        causal_mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        maskedScores = scaledScores.masked_fill(causal_mask_bool, -torch.inf)

        preDropoutWeights = torch.softmax(maskedScores, dim=-1)
        print(f"Pre-Dropout Causal Attention Weights (rows sum to 1.0):\n{preDropoutWeights}\n")

        print("=" * 70)
        print(f"STEP 4: Apply Attention Dropout (p = {self.dropout_p}, training={self.training})")
        print("=" * 70)
        scale_factor = 1.0 / (1.0 - self.dropout_p) if (self.training and self.dropout_p < 1.0) else 1.0
        print("How PyTorch's Inverted Dropout works:")
        print(f"  - Each weight is zeroed out independently with probability p = {self.dropout_p}.")
        print(f"  - Surviving weights are multiplied by 1 / (1 - {self.dropout_p}) = {scale_factor:.4f}")
        print("    so that expected values match between training and inference (eval mode).\n")

        attentionWeights = self.dropout(preDropoutWeights)
        print(f"Post-Dropout Attention Weights, shape {tuple(attentionWeights.shape)}:")
        print(f"{attentionWeights}\n")
        print(f"Row sums after dropout (fluctuate around 1.0 due to random sampling):\n"
              f"  {attentionWeights.sum(dim=-1)}\n")

        print("=" * 70)
        print("STEP 5: Context Vector = Post-Dropout Attention Weights @ Values")
        print("=" * 70)
        context_vector = attentionWeights @ values
        print(f"context_vector, shape {tuple(context_vector.shape)}:\n{context_vector}\n")

        if self.visualize and not is_batched:
            plot_dropout_pipeline(
                self.word_labels, preDropoutWeights, attentionWeights, self.dropout_p
            )
            plot_matrix_multiplication(
                attentionWeights, values, context_vector,
                A_label=f"Post-Dropout Weights (p={self.dropout_p})",
                B_label="Values",
                C_label="Context Vector",
                A_row_labels=self.word_labels,
                C_row_labels=self.word_labels,
            )

        self.last_queries = queries
        self.last_keys = keys
        self.last_values = values
        self.last_attention_scores = attentionScores
        self.last_masked_scores = maskedScores
        self.last_pre_dropout_weights = preDropoutWeights
        self.last_attention_weights = attentionWeights

        return context_vector


if __name__ == "__main__":
    args = parse_args()
    input_words = re.findall(r"\b[\w']+\b", args.sentence)

    print("Sentence Used:", args.sentence)
    print("Words:", input_words)

    # ------------------------------------------------------------------
    # Warm-up: Sebastian Raschka's toy ones-matrix demonstration so the
    # 1 / (1 - p) rescaling is crystal clear before looking at real floats
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"WARM-UP: What nn.Dropout(p={args.dropout}) does to a 6x6 matrix of 1s")
    print("=" * 70)
    torch.manual_seed(123)
    toy_dropout = torch.nn.Dropout(args.dropout)
    ones_example = torch.ones(6, 6)
    print(f"Before dropout:\n{ones_example}\n")
    print(f"After nn.Dropout(p={args.dropout}) (notice survivors are scaled to "
          f"1/(1-{args.dropout}) = {1.0 / (1.0 - args.dropout):.2f}):\n{toy_dropout(ones_example)}\n")

    # ------------------------------------------------------------------
    # Main Walkthrough on the sentence's 6D Word2Vec embeddings
    # ------------------------------------------------------------------
    inputs = generate_semantic_embeddings(input_words, output_dim=6)
    print("Embeddings (from word2vec, reduced to 6D):\n", inputs)

    context_length = inputs.shape[0]
    d_in = inputs.shape[1]
    d_out = inputs.shape[1] - 1

    torch.manual_seed(15033019)
    causal_attention = CausalAttention(
        d_in, d_out, context_length=context_length, dropout=args.dropout,
        visualize=True, word_labels=input_words
    )
    if args.pca_init:
        causal_attention.init_qkv_from_pca(inputs)

    # 1. Run in TRAINING mode (default: causal_attention.training == True)
    causal_attention.train()
    torch.manual_seed(42)  # reproducible dropout mask
    context_vector_train = causal_attention(inputs)
    train_weights = causal_attention.last_attention_weights.clone()
    pre_weights = causal_attention.last_pre_dropout_weights.clone()

    print_dropout_summary(input_words, pre_weights, train_weights, args.dropout)

    # 2. Compare with EVALUATION / INFERENCE mode (causal_attention.eval())
    print("\n" + "=" * 70)
    print("BONUS STEP 6: Training Mode (model.train()) vs Inference Mode (model.eval())")
    print("=" * 70)
    causal_attention.eval()
    with torch.no_grad():
        # Temporarily turn off step-by-step plots during the eval pass
        causal_attention.visualize = False
        context_vector_eval = causal_attention(inputs)
        eval_weights = causal_attention.last_attention_weights.clone()

    plot_train_vs_eval_comparison(
        input_words, train_weights, eval_weights, args.dropout
    )

    # 3. Also show batched 3D input support (batch_size=2, num_tokens, d_in)
    # exactly as used when plugging into the DataLoader from File 3!
    print("=" * 70)
    print("BONUS STEP 7: Handling a 3D Batch (batch_size=2, num_tokens, d_in)")
    print("=" * 70)
    batch = torch.stack((inputs, inputs), dim=0)
    print(f"Batched input shape : {tuple(batch.shape)}  (2 sentences x {context_length} tokens x {d_in} dims)")
    causal_attention.train()
    torch.manual_seed(42)
    batch_context = causal_attention(batch)
    print(f"Batched output shape: {tuple(batch_context.shape)}  (2 sentences x {context_length} tokens x {d_out} dims)")
    print("(Notice that Batch 0 and Batch 1 receive independent random dropout masks during training!)")
