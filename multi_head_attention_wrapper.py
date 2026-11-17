# Ramakrishnan A
# File 10-A in the series on Building LLMs from scratch
# Multi-Head Attention via Stacking (`MultiHeadAttentionWrapper`): running
# multiple independent CausalAttention heads in parallel and concatenating
# their context vectors along the feature dimension.
# Lots of pranams to Sebastian Raschka

import torch.nn as nn
import torch
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Reuse the word2vec-backed embedding generator from File 6 and the
# CausalAttention building block from File 9-B.
from attention_without_trainable_weights import generate_semantic_embeddings
from causal_attention_with_dropout import CausalAttention


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Head Causal Self-Attention built by stacking multiple "
                    "single-head CausalAttention modules (MultiHeadAttentionWrapper)."
    )
    parser.add_argument("sentence", type=str, help="The sentence to tokenize and embed")
    parser.add_argument("--num-heads", type=int, default=2,
                        help="Number of parallel CausalAttention heads to stack (default: 2).")
    parser.add_argument("--head-dim", type=int, default=2,
                        help="Output dimension d_out of each individual head (default: 2, "
                             "so 2 heads produce a 2*2 = 4D concatenated context vector).")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Attention dropout probability p in [0, 1) (default: 0.0 so "
                             "per-head attention patterns are clean to compare; pass e.g. "
                             "--dropout 0.2 to see dropout active across heads).")
    parser.add_argument("--pca-init", action="store_true",
                        help="Initialize each head with a distinct orthogonal slice of PCA "
                             "principal components of the input embeddings (Head 0 gets PCs "
                             "0..d_out-1, Head 1 gets PCs d_out..2*d_out-1, etc.) so heads "
                             "specialize in different semantic directions without training.")
    return parser.parse_args()


def plot_multi_head_heatmaps(input_words, head_weights_list, figsize_per_head=(5.5, 5),
                             save_path=None, show=True):
    """
    Plot each head's causal attention weight matrix side-by-side on a shared
    color scale, plus a final panel showing the absolute difference between
    Head 1 and Head 2 so you can immediately see where the heads disagree.
    """
    num_heads = len(head_weights_list)
    weights_np = [w.detach().numpy() for w in head_weights_list]
    n = len(input_words)
    vmax = max(max(w.max() for w in weights_np), 1e-6)

    show_diff = (num_heads >= 2)
    total_panels = num_heads + (1 if show_diff else 0)
    fig, axes = plt.subplots(
        1, total_panels,
        figsize=(figsize_per_head[0] * total_panels, figsize_per_head[1])
    )
    if total_panels == 1:
        axes = [axes]

    for h_idx in range(num_heads):
        ax = axes[h_idx]
        mat = weights_np[h_idx]
        im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
        ax.set_title(f"Head {h_idx + 1} Causal Attention Weights\nshape {mat.shape}", fontsize=10.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(input_words, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(input_words if h_idx == 0 else [], fontsize=8)
        if h_idx == 0:
            ax.set_ylabel("Query (attending from)")
        ax.set_xlabel("Key (attended to)")

        if n <= 12:
            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    col = 'white' if val < vmax * 0.6 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7, color=col)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_diff:
        ax_d = axes[-1]
        diff_mat = weights_np[0] - weights_np[1]
        max_abs = max(np.abs(diff_mat).max(), 1e-6)
        im_d = ax_d.imshow(diff_mat, cmap='coolwarm', vmin=-max_abs, vmax=max_abs, aspect='auto')
        ax_d.set_title("Head 1 − Head 2 Divergence\n(Red = Head 1 higher, Blue = Head 2 higher)", fontsize=10.5)
        ax_d.set_xticks(range(n))
        ax_d.set_yticks(range(n))
        ax_d.set_xticklabels(input_words, rotation=45, ha='right', fontsize=8)
        ax_d.set_yticklabels([], fontsize=8)
        ax_d.set_xlabel("Key (attended to)")
        if n <= 12:
            for i in range(n):
                for j in range(n):
                    val = diff_mat[i, j]
                    col = 'white' if abs(val) > max_abs * 0.55 else 'black'
                    txt = f"{val:+.2f}" if j <= i else "0.00"
                    ax_d.text(j, i, txt, ha='center', va='center', fontsize=7, color=col)
        fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Multi-Head Causal Attention: {num_heads} Independent Heads Attending to Different Subspaces",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_head_concatenation(input_words, head_outputs, combined_output,
                            cmap='coolwarm', figsize=(15, 5.5),
                            save_path=None, show=True):
    """
    Visualize torch.cat([head_1(x), head_2(x), ...], dim=-1):
    shows each head's individual (num_tokens, d_out) context matrix connected
    by '||' concatenation symbols to the final (num_tokens, num_heads * d_out)
    multi-head context matrix, with dashed vertical lines marking head boundaries.
    """
    heads_np = [h.detach().numpy() for h in head_outputs]
    comb_np = combined_output.detach().numpy()
    num_heads = len(heads_np)
    d_out = heads_np[0].shape[1]

    # Build width ratios: Head_1 | "||" | Head_2 | ... | "=" | Combined
    widths = []
    for h in heads_np:
        widths.append(max(h.shape[1], 1.2))
        widths.append(0.55)
    widths[-1] = 0.7  # last separator before Combined is '='
    widths.append(max(comb_np.shape[1], 2.0))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, len(widths), width_ratios=widths, figure=fig)
    vmax = max(np.abs(comb_np).max(), 1e-6)

    def draw_mat(ax, M, title, row_labels=None, col_labels=None):
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f"{title}\nshape {M.shape}", fontsize=10)
        ax.set_xticks(range(M.shape[1]))
        if col_labels is not None:
            ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(M.shape[0]))
        if row_labels is not None:
            ax.set_yticklabels(row_labels, fontsize=8)
        else:
            ax.set_yticklabels([])
        if M.shape[0] * M.shape[1] <= 160:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    col = 'white' if abs(M[i, j]) > vmax * 0.6 else 'black'
                    ax.text(j, i, f"{M[i, j]:.2f}", ha='center', va='center',
                            fontsize=6.5, color=col)
        return im

    for h_idx, h_mat in enumerate(heads_np):
        ax_h = fig.add_subplot(gs[0, 2 * h_idx])
        col_lbls = [f"h{h_idx + 1}_{d}" for d in range(h_mat.shape[1])]
        draw_mat(
            ax_h, h_mat, f"Head {h_idx + 1} Context",
            row_labels=input_words if h_idx == 0 else None,
            col_labels=col_lbls,
        )

        ax_sep = fig.add_subplot(gs[0, 2 * h_idx + 1])
        ax_sep.axis('off')
        symbol = '‖' if h_idx < num_heads - 1 else '='
        subtitle = 'cat(dim=-1)' if h_idx < num_heads - 1 else ''
        ax_sep.text(0.5, 0.52, symbol, ha='center', va='center', fontsize=26, fontweight='bold')
        if subtitle:
            ax_sep.text(0.5, 0.36, subtitle, ha='center', va='center', fontsize=8, color='#555555')

    ax_c = fig.add_subplot(gs[0, -1])
    comb_col_lbls = [
        f"h{h_idx + 1}_{d}"
        for h_idx in range(num_heads)
        for d in range(d_out)
    ]
    draw_mat(
        ax_c, comb_np,
        f"Concatenated Multi-Head Context\n(num_heads={num_heads} × d_out={d_out} = {comb_np.shape[1]} cols)",
        row_labels=input_words,
        col_labels=comb_col_lbls,
    )
    # Draw dashed vertical lines between heads inside the concatenated matrix
    for boundary in range(1, num_heads):
        ax_c.axvline(boundary * d_out - 0.5, color='yellow', linestyle='--', linewidth=1.8)

    plt.suptitle("How MultiHeadAttentionWrapper Concatenates Per-Head Context Vectors Along dim=-1",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def print_multi_head_comparison(input_words, head_weights_list):
    """
    Compare what each head attends to for every word in the sentence, plus
    each head's average causal Shannon entropy.
    """
    num_heads = len(head_weights_list)
    weights_np = [w.detach().numpy() for w in head_weights_list]

    print("\n" + "=" * 78)
    print(f"PER-WORD COMPARISON ACROSS ALL {num_heads} HEADS (Top-2 Attended Visible Words)")
    print("=" * 78)
    header = f"{'Pos':<5}{'Word':<12}" + "".join([f"{'Head ' + str(h + 1) + ' Top Attended':<30}" for h in range(num_heads)])
    print(header)
    print("-" * len(header))

    for i, word in enumerate(input_words):
        row_str = f"{i:<5}{word:<12}"
        for h_idx in range(num_heads):
            visible_row = weights_np[h_idx][i, :i + 1]
            top_indices = visible_row.argsort()[::-1][:min(2, i + 1)]
            parts = [f"{input_words[idx]} ({visible_row[idx]:.2f})" for idx in top_indices]
            row_str += f"{', '.join(parts):<30}"
        print(row_str)

    print("\nPer-Head Causal Entropy Summary:")
    for h_idx in range(num_heads):
        entropies = []
        for i in range(len(input_words)):
            vis = weights_np[h_idx][i, :i + 1]
            ent = max(0.0, float(-np.sum(vis * np.log(vis + 1e-12))))
            entropies.append(ent)
        print(f"  Head {h_idx + 1} mean causal entropy: {np.mean(entropies):.4f}")

    if num_heads >= 2:
        mean_l1_diff = np.mean(np.abs(weights_np[0] - weights_np[1]))
        print(f"  Mean |Head 1 - Head 2| attention weight difference: {mean_l1_diff:.4f}")


class MultiHeadAttentionWrapper(nn.Module):
    """
    Multi-Head Causal Attention built by stacking multiple single-head
    `CausalAttention` modules in an `nn.ModuleList` (matching Sebastian
    Raschka's `MultiHeadAttentionWrapper` in Chapter 3.6.1).

    Why we need multiple heads:
      A single CausalAttention head has only ONE set of (WQuery, WKey, WValue)
      matrices, so each word only gets a single query vector to probe previous
      words. By running `num_heads` independent CausalAttention heads in
      parallel, each head can project the input embeddings into a different
      representation subspace (e.g., Head 1 focusing on subject-verb links,
      Head 2 focusing on pronoun-noun coreference or nearby adjectives).

    How their outputs are combined here:
      If each head produces a context vector of shape `(num_tokens, d_out)`,
      concatenating `num_heads` of them along the last dimension (`dim=-1`)
      produces a combined context vector of shape:
        `(num_tokens, num_heads * d_out)`  (or `(batch, num_tokens, num_heads * d_out)`).
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False, visualize=False, word_labels=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.visualize = visualize
        self.word_labels = word_labels

        # Create `num_heads` independent CausalAttention instances. Because
        # PyTorch's random number generator advances after each head's
        # nn.Linear layers are constructed, every head starts with DIFFERENT
        # initial WQuery, WKey, WValue weights!
        self.heads = nn.ModuleList([
            CausalAttention(
                d_in=d_in,
                d_out=d_out,
                context_length=context_length,
                dropout=dropout,
                qkv_bias=qkv_bias,
                visualize=False,  # we do clean multi-head visuals at the wrapper level
                word_labels=word_labels,
            )
            for _ in range(num_heads)
        ])

        # Stashed after forward() for inspection and plotting
        self.last_head_outputs = []
        self.last_head_attention_weights = []

    def init_heads_from_pca_subspaces(self, x):
        """
        Initialize each head's (WQuery, WKey, WValue) using a DISTINCT
        orthogonal slice of principal components from the input embeddings X.

        Why distinct slices?
          If every head were initialized with the exact same top `d_out`
          principal components, all heads would compute identical Q, K, V
          and identical attention weights! Instead:
            - Head 1 gets PCs [0 .. d_out - 1]
            - Head 2 gets PCs [d_out .. 2*d_out - 1]
            - Head k gets PCs [(k-1)*d_out .. k*d_out - 1] (wrapping if needed)
          so each head inspects an orthogonal semantic subspace of the
          Word2Vec embeddings right from step 0.
        """
        with torch.no_grad():
            x_np = x.detach().numpy()
            centered = x_np - x_np.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            total_pcs = Vt.shape[0]

            for h_idx, head in enumerate(self.heads):
                indices = [(h_idx * self.d_out + j) % total_pcs for j in range(self.d_out)]
                pca_slice = torch.tensor(Vt[indices], dtype=torch.float32)
                head.WQuery.weight.copy_(pca_slice)
                head.WKey.weight.copy_(pca_slice)
                head.WValue.weight.copy_(pca_slice)
                print(f"  Initialized Head {h_idx + 1} WQuery/WKey/WValue from PCA components {indices}")

    def forward(self, x):
        is_batched = (x.dim() == 3)

        print("=" * 78)
        print(f"STEP 1: Run each of the {self.num_heads} CausalAttention heads independently on Input x")
        print("=" * 78)
        print(f"Input x shape: {tuple(x.shape)}  |  num_heads = {self.num_heads}  |  per-head d_out = {self.d_out}")
        print(f"Expected concatenated output last dim = num_heads * d_out = {self.num_heads * self.d_out}\n")

        head_outputs = []
        head_weights = []

        for h_idx, head in enumerate(self.heads):
            print("-" * 78)
            print(f">>> Running Head {h_idx + 1} of {self.num_heads}")
            print("-" * 78)
            print(f"Head {h_idx + 1} WQuery.weight, shape {tuple(head.WQuery.weight.shape)}:\n{head.WQuery.weight.data}\n")

            # Run the underlying CausalAttention head (from File 9-B)
            out_h = head(x)
            head_outputs.append(out_h)
            head_weights.append(head.last_attention_weights)

            print(f"Head {h_idx + 1} output context vector, shape {tuple(out_h.shape)}:\n{out_h}\n")

        print("=" * 78)
        print(f"STEP 2: Concatenate all {self.num_heads} head outputs along the last dimension (dim=-1)")
        print("=" * 78)
        shapes_str = " || ".join([str(tuple(h.shape)) for h in head_outputs])
        context_vec = torch.cat(head_outputs, dim=-1)
        print(f"Concatenating: {shapes_str}  -->  {tuple(context_vec.shape)}\n")
        print(f"Combined Multi-Head Context Vector, shape {tuple(context_vec.shape)}:\n{context_vec}\n")

        if self.visualize and not is_batched:
            plot_multi_head_heatmaps(self.word_labels, head_weights)
            plot_head_concatenation(self.word_labels, head_outputs, context_vec)

        self.last_head_outputs = head_outputs
        self.last_head_attention_weights = head_weights
        return context_vec


if __name__ == "__main__":
    args = parse_args()
    input_words = re.findall(r"\b[\w']+\b", args.sentence)

    print("Sentence Used:", args.sentence)
    print("Words:", input_words)

    # Generate 6D semantic embeddings from word2vec (matching Files 7-D, 9-A, 9-B)
    embed_dim = max(6, args.num_heads * args.head_dim)
    inputs = generate_semantic_embeddings(input_words, output_dim=embed_dim)
    print(f"Embeddings (from word2vec, reduced to {inputs.shape[1]}D):\n", inputs)

    context_length = inputs.shape[0]
    d_in = inputs.shape[1]
    d_out = args.head_dim

    torch.manual_seed(15033019)
    mha_wrapper = MultiHeadAttentionWrapper(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=args.dropout,
        num_heads=args.num_heads,
        visualize=True,
        word_labels=input_words,
    )

    if args.pca_init:
        print("\nInitializing each head from a distinct orthogonal PCA subspace:")
        mha_wrapper.init_heads_from_pca_subspaces(inputs)
        print()

    # 1. Run on the 2D sentence embeddings (num_tokens, d_in)
    context_vecs = mha_wrapper(inputs)

    print_multi_head_comparison(input_words, mha_wrapper.last_head_attention_weights)

    # 2. Also demonstrate 3D batched input (batch_size=2, num_tokens, d_in),
    # matching Sebastian Raschka's Chapter 3.6.1 example!
    print("\n" + "=" * 78)
    print("BONUS STEP 3: Batched 3D Input (batch_size=2, num_tokens, d_in)")
    print("=" * 78)
    batch = torch.stack((inputs, inputs), dim=0)
    mha_wrapper.visualize = False
    batched_context_vecs = mha_wrapper(batch)
    print(f"Batched input shape : {tuple(batch.shape)}")
    print(f"Batched output shape: {tuple(batched_context_vecs.shape)} "
          f"(batch=2, num_tokens={context_length}, num_heads*d_out={args.num_heads * d_out})")
