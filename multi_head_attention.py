# Ramakrishnan A
# File 10-B in the series on Building LLMs from scratch
# Production Multi-Head Attention with Weight Splits (`MultiHeadAttention`):
# computing all attention heads in a single batched matrix multiplication
# via `.view(...)` and `.transpose(...)`, followed by `out_proj` (W_O).
# Lots of pranams to Sebastian Raschka

import torch.nn as nn
import torch
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Reuse the word2vec-backed embedding generator from File 6 and the
# MultiHeadAttentionWrapper + comparison/plotting helpers from File 10-A.
from attention_without_trainable_weights import generate_semantic_embeddings
from multi_head_attention_wrapper import (
    MultiHeadAttentionWrapper,
    plot_multi_head_heatmaps,
    print_multi_head_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production Multi-Head Causal Self-Attention using a single "
                    "batched Q/K/V projection split across heads + output projection W_O."
    )
    parser.add_argument("sentence", type=str, help="The sentence to tokenize and embed")
    parser.add_argument("--num-heads", type=int, default=2,
                        help="Number of attention heads (default: 2).")
    parser.add_argument("--d-out", type=int, default=4,
                        help="Total output dimension across all heads (must be divisible by "
                             "--num-heads; default: 4, so head_dim = 4/2 = 2).")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Attention dropout probability p in [0, 1) (default: 0.0).")
    parser.add_argument("--qkv-bias", action="store_true",
                        help="Include learnable bias terms in WQuery, WKey, WValue.")
    parser.add_argument("--pca-init", action="store_true",
                        help="Initialize each head's slice of WQuery/WKey/WValue from a "
                             "distinct orthogonal slice of PCA principal components.")
    return parser.parse_args()


def plot_out_proj_mixing(input_words, pre_proj_context, out_proj_weight, final_output,
                         cmap='coolwarm', figsize=(16, 5.5),
                         save_path=None, show=True):
    """
    Visualize how `out_proj` (W_O) mixes information across all attention heads:
      Pre-Projection Context (T x d_out)  @  W_O.T (d_out x d_out)  =  Final Output (T x d_out)
    """
    A = pre_proj_context.detach().numpy()
    B = out_proj_weight.T.detach().numpy()
    C = final_output.detach().numpy()

    widths = [max(A.shape[1], 1.5), 0.55, max(B.shape[1], 1.5), 0.55, max(C.shape[1], 1.5)]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 5, width_ratios=widths, figure=fig)

    def draw_mat(ax, M, title, row_labels=None):
        vmax = max(np.abs(M).max(), 1e-6)
        ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f"{title}\nshape {M.shape}", fontsize=10)
        ax.set_xticks(range(M.shape[1]))
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

    ax_a = fig.add_subplot(gs[0, 0])
    draw_mat(ax_a, A, "Concatenated Heads\n(Before out_proj)", row_labels=input_words)

    ax_x = fig.add_subplot(gs[0, 1]); ax_x.axis('off')
    ax_x.text(0.5, 0.5, '×', ha='center', va='center', fontsize=28)

    ax_b = fig.add_subplot(gs[0, 2])
    draw_mat(ax_b, B, "out_proj.weight.T (W_O)")

    ax_eq = fig.add_subplot(gs[0, 3]); ax_eq.axis('off')
    ax_eq.text(0.5, 0.5, '=', ha='center', va='center', fontsize=28)

    ax_c = fig.add_subplot(gs[0, 4])
    draw_mat(ax_c, C, "Final MHA Output\n(Heads Mixed by W_O)", row_labels=input_words)

    plt.suptitle("How out_proj (W_O) Mixes Information Across Attention Heads",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


class MultiHeadAttention(nn.Module):
    """
    Production Multi-Head Causal Self-Attention with Weight Splits (matching
    Sebastian Raschka's `MultiHeadAttention` in Chapter 3.6.2).

    Key differences from `MultiHeadAttentionWrapper` (File 10-A):
      1. Instead of looping over `num_heads` separate `CausalAttention` layers,
         we use ONE `WQuery`, ONE `WKey`, and ONE `WValue` of size `(d_in, d_out)`
         where `d_out = num_heads * head_dim`.
      2. After projecting `x`, we reshape `.view(b, num_tokens, num_heads, head_dim)`
         and `.transpose(1, 2)` to `(b, num_heads, num_tokens, head_dim)` so all
         heads run in a SINGLE parallel matrix multiplication.
      3. After merging the heads back to `(b, num_tokens, d_out)`, we apply a
         final linear projection `self.out_proj` (W_O) so features learned by
         different heads can interact and mix.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False, visualize=False, word_labels=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # reduce projection dim to match desired output dim
        self.context_length = context_length
        self.dropout_p = dropout
        self.visualize = visualize
        self.word_labels = word_labels

        self.WQuery = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WKey = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WValue = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer (W_O) to combine head outputs
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # Populated during forward() for inspection/visualization
        self.last_queries = None
        self.last_keys = None
        self.last_values = None
        self.last_attention_scores = None
        self.last_attention_weights = None
        self.last_pre_proj_context = None
        self.last_output = None

    def init_qkv_from_pca_subspaces(self, x):
        """
        Initialize each head's row slice `[h * head_dim : (h + 1) * head_dim]`
        inside the unified `WQuery`, `WKey`, `WValue` matrices from distinct
        orthogonal PCA components of `x`.
        """
        with torch.no_grad():
            x_np = x.detach().numpy()
            centered = x_np - x_np.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            total_pcs = Vt.shape[0]

            for h_idx in range(self.num_heads):
                indices = [(h_idx * self.head_dim + j) % total_pcs for j in range(self.head_dim)]
                pca_slice = torch.tensor(Vt[indices], dtype=torch.float32)
                start = h_idx * self.head_dim
                end = start + self.head_dim
                self.WQuery.weight[start:end].copy_(pca_slice)
                self.WKey.weight[start:end].copy_(pca_slice)
                self.WValue.weight[start:end].copy_(pca_slice)
                print(f"  Initialized head {h_idx + 1} rows [{start}:{end}] from PCA components {indices}")

    def forward(self, x):
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        b, num_tokens, d_in = x.shape

        print("=" * 78)
        print("STEP 1: Unified Q, K, V Projections (One Matrix Multiply Each)")
        print("=" * 78)
        keys = self.WKey(x)      # shape: (b, num_tokens, d_out)
        queries = self.WQuery(x) # shape: (b, num_tokens, d_out)
        values = self.WValue(x)  # shape: (b, num_tokens, d_out)
        print(f"Input x shape        : {tuple(x.shape)}")
        print(f"Projected Q/K/V shape: {tuple(queries.shape)}  (d_out = {self.d_out} = {self.num_heads} heads × {self.head_dim} head_dim)\n")

        print("=" * 78)
        print("STEP 2: Split d_out -> (num_heads, head_dim) and Transpose to 4D")
        print("=" * 78)
        # Implicitly split the matrix by adding a `num_heads` dimension:
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        print(f"After .view(b, T, num_heads, head_dim)  : {tuple(queries.shape)}")

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        print(f"After .transpose(1, 2)                  : {tuple(queries.shape)}  (batch, heads, tokens, head_dim)\n")

        print("=" * 78)
        print("STEP 3: Batched Scaled Causal Attention Across All Heads at Once")
        print("=" * 78)
        attn_scores = queries @ keys.transpose(2, 3)  # dot product for each head: (b, num_heads, num_tokens, num_tokens)
        print(f"attn_scores = queries @ keys.transpose(2, 3), shape {tuple(attn_scores.shape)}")

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        print(f"attn_weights (after -inf mask, /sqrt({self.head_dim}), softmax, dropout={self.dropout_p}), shape {tuple(attn_weights.shape)}\n")

        print("=" * 78)
        print("STEP 4: Weighted Values & Merge Heads Back to (b, num_tokens, d_out)")
        print("=" * 78)
        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        print(f"(attn_weights @ values).transpose(1, 2) shape: {tuple(context_vec.shape)}")

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        print(f"After .contiguous().view(b, T, d_out)        : {tuple(context_vec.shape)}\n")

        print("=" * 78)
        print("STEP 5: Final Output Projection `out_proj` (W_O)")
        print("=" * 78)
        out = self.out_proj(context_vec)
        print(f"out = self.out_proj(context_vec), shape {tuple(out.shape)}:\n{out}\n")

        self.last_queries = queries
        self.last_keys = keys
        self.last_values = values
        self.last_attention_scores = attn_scores
        self.last_attention_weights = attn_weights
        self.last_pre_proj_context = context_vec.squeeze(0) if squeeze_batch else context_vec
        out_final = out.squeeze(0) if squeeze_batch else out
        self.last_output = out_final

        if self.visualize and squeeze_batch:
            head_weights_list = [attn_weights[0, h] for h in range(self.num_heads)]
            plot_multi_head_heatmaps(self.word_labels, head_weights_list)
            plot_out_proj_mixing(
                self.word_labels,
                self.last_pre_proj_context,
                self.out_proj.weight,
                out_final,
            )

        return out_final


def verify_equivalence_with_wrapper(mha, inputs, input_words):
    """
    Prove that `MultiHeadAttention` (File 10-B) computes the EXACT same
    pre-`out_proj` context vectors as `MultiHeadAttentionWrapper` (File 10-A)
    when given the exact same weights, just in one batched matrix multiply
    instead of a Python loop over heads.
    """
    print("=" * 78)
    print("BONUS STEP 6: Equivalence Check vs. MultiHeadAttentionWrapper (File 10-A)")
    print("=" * 78)

    wrapper = MultiHeadAttentionWrapper(
        d_in=mha.d_in,
        d_out=mha.head_dim,
        context_length=mha.context_length,
        dropout=0.0,  # disable random dropout for exact numerical comparison
        num_heads=mha.num_heads,
        qkv_bias=(mha.WQuery.bias is not None),
        visualize=False,
        word_labels=input_words,
    )

    with torch.no_grad():
        for h_idx, head in enumerate(wrapper.heads):
            start = h_idx * mha.head_dim
            end = start + mha.head_dim
            head.WQuery.weight.copy_(mha.WQuery.weight[start:end])
            head.WKey.weight.copy_(mha.WKey.weight[start:end])
            head.WValue.weight.copy_(mha.WValue.weight[start:end])
            if mha.WQuery.bias is not None:
                head.WQuery.bias.copy_(mha.WQuery.bias[start:end])
                head.WKey.bias.copy_(mha.WKey.bias[start:end])
                head.WValue.bias.copy_(mha.WValue.bias[start:end])

    # Run wrapper silently to compare tensors cleanly
    wrapper.eval()
    mha_was_training = mha.training
    mha_was_vis = mha.visualize
    mha.eval()
    mha.visualize = False

    with torch.no_grad():
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            wrapper_out = wrapper(inputs)
            _ = mha(inputs)

    max_diff = torch.max(torch.abs(wrapper_out - mha.last_pre_proj_context)).item()
    print(f"Max absolute difference between File 10-A Wrapper and File 10-B Pre-Proj Context: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Expected exact match, got {max_diff}"
    print("PASS: Weight-split MultiHeadAttention matches stacked MultiHeadAttentionWrapper!\n")

    mha.visualize = mha_was_vis
    if mha_was_training:
        mha.train()


if __name__ == "__main__":
    args = parse_args()
    input_words = re.findall(r"\b[\w']+\b", args.sentence)

    print("Sentence Used:", args.sentence)
    print("Words:", input_words)

    embed_dim = max(6, args.d_out)
    inputs = generate_semantic_embeddings(input_words, output_dim=embed_dim)
    print(f"Embeddings (from word2vec, reduced to {inputs.shape[1]}D):\n", inputs)

    context_length = inputs.shape[0]
    d_in = inputs.shape[1]

    torch.manual_seed(15033019)
    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=args.d_out,
        context_length=context_length,
        dropout=args.dropout,
        num_heads=args.num_heads,
        qkv_bias=args.qkv_bias,
        visualize=True,
        word_labels=input_words,
    )

    if args.pca_init:
        print("\nInitializing unified Q/K/V slices from distinct orthogonal PCA subspaces:")
        mha.init_qkv_from_pca_subspaces(inputs)
        print()

    # 1. Run on 2D sentence embeddings (num_tokens, d_in)
    context_vecs = mha(inputs)

    head_weights_list = [mha.last_attention_weights[0, h] for h in range(mha.num_heads)]
    print_multi_head_comparison(input_words, head_weights_list)

    # 2. Prove mathematical equivalence with File 10-A's MultiHeadAttentionWrapper
    verify_equivalence_with_wrapper(mha, inputs, input_words)

    # 3. Demonstrate 3D batched input (batch_size=2, num_tokens, d_in)
    print("=" * 78)
    print("BONUS STEP 7: Batched 3D Input (batch_size=2, num_tokens, d_in)")
    print("=" * 78)
    batch = torch.stack((inputs, inputs), dim=0)
    mha.visualize = False
    batched_out = mha(batch)
    print(f"Batched input shape : {tuple(batch.shape)}")
    print(f"Batched output shape: {tuple(batched_out.shape)} "
          f"(batch=2, num_tokens={context_length}, d_out={args.d_out})")
