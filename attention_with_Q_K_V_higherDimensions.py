# Ramakrishnan A
# File 7-D in the series on Building LLMs from scratch
# Self-attention with trainable Q, K, V weights on higher-dimensional (6D -> 5D) embeddings + PCA init
# Lots of pranams to Sebastian Raschka

import torch.nn as nn
import torch
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Reuse the word2vec-backed embedding generator (with local caching) from
# File 6, rather than duplicating it here. This works because that file's
# execution code is wrapped in `if __name__ == "__main__":`, so importing
# it only pulls in the function/constant definitions -- no side effects.
from attention_without_trainable_weights import generate_semantic_embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-attention with trainable Q/K/V weights, using word2vec "
                     "embeddings for the input sentence."
    )
    parser.add_argument("sentence", type=str, help="The sentence to tokenize and embed")
    parser.add_argument("--pca-init", action="store_true",
                         help="Initialize WQuery/WKey/WValue from PCA of the input "
                              "embeddings instead of random weights -- no training, "
                              "but tends to produce more semantically meaningful "
                              "attention than a random projection at low d_out.")
    return parser.parse_args()

'''
input_words = ["the", "dog", "chased", 'the', 'cat', 'because', 'it', 'was', 'very', 'hungry']

inputs = torch.tensor([
    [0.10, 0.10, 0.10], #The
    [0.85, 0.75, 0.65], #dog
    [0.70, 0.65, 0.75], #chased
    [0.10, 0.10, 0.10], #the
    [0.78, 0.70, 0.62], #cat
    [0.30, 0.25, 0.35], #because
    [0.72, 0.68, 0.60], #it
    [0.25, 0.20, 0.30], #was
    [0.35, 0.40, 0.30], #very
    [0.65, 0.75, 0.55]  #hungry
])

d_in = inputs.shape[1]
d_out = inputs.shape[1] - 1
'''


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


def compute_attention_entropy(input_words, attention_weights):
    """
    Compute and print the Shannon entropy of each word's attention
    distribution. Max possible entropy = log(n) means completely uniform
    (no preference at all); lower entropy means attention is more focused
    on specific words.
    """
    weights_np = attention_weights.detach().numpy()
    n = weights_np.shape[1]
    max_entropy = np.log(n)

    print(f"\nAttention Entropy per word (max possible = {max_entropy:.4f}, i.e. completely uniform):")
    print(f"{'Word':<12}{'Entropy':<12}{'% of Max':<10}")
    print("-" * 34)

    entropies = []
    for word, row in zip(input_words, weights_np):
        entropy = -np.sum(row * np.log(row + 1e-12))
        entropies.append(entropy)
        pct = 100 * entropy / max_entropy
        print(f"{word:<12}{entropy:<12.4f}{pct:<10.1f}")

    avg_entropy = np.mean(entropies)
    print(f"\nAverage entropy: {avg_entropy:.4f} ({100 * avg_entropy / max_entropy:.1f}% of max)")
    return entropies


def print_top_k_attended(input_words, attention_weights, k=3):
    """
    Print, in plain English, which words each word attends to most.
    """
    weights_np = attention_weights.detach().numpy()
    print(f"\nTop-{k} attended words per word:")
    for i, word in enumerate(input_words):
        row = weights_np[i]
        top_indices = row.argsort()[::-1][:k]
        parts = [f"{input_words[idx]} ({row[idx]:.2f})" for idx in top_indices]
        print(f'  "{word}" attends most to: ' + ", ".join(parts))


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
        # annotate with actual values when the matrix is small enough to read
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


class SelfAttentionv1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False, visualize=False, word_labels=None):
        super().__init__()
        self.visualize = visualize      # if True, plot each matrix multiplication step
        self.word_labels = word_labels  # optional row labels for the plots
        '''
        self.WQuery = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.WKey = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.WValue = torch.nn.Parameter(torch.rand(d_in, d_out))
        '''
        self.WQuery = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WKey = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WValue = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        # populated by forward() so they can be inspected/visualized afterward
        self.last_queries = None
        self.last_keys = None
        self.last_values = None
        self.last_attention_scores = None
        self.last_attention_weights = None

    def init_qkv_from_pca(self, x):
        """
        Initialize WQuery, WKey, WValue using the top principal components
        of the actual input embeddings -- pure linear algebra (SVD), no
        gradient training involved.

        Why this helps: at low output dimensions (e.g. d_out=2), a random
        projection tends to scramble whatever real semantic structure
        already exists in the input embeddings (like word2vec vectors),
        often collapsing attention onto whichever word happens to get an
        oversized random vector rather than genuinely related words. PCA
        instead picks the directions of greatest variance in the data --
        the directions most likely to separate genuinely different words
        -- so the resulting attention pattern reflects real structure
        instead of an initialization artifact.
        """
        with torch.no_grad():
            x_np = x.detach().numpy()
            centered = x_np - x_np.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            d_out = self.WQuery.weight.shape[0]
            pca_components = Vt[:d_out]  # shape (d_out, d_in) -- matches nn.Linear.weight directly
            pca_tensor = torch.tensor(pca_components, dtype=torch.float32)
            self.WQuery.weight.copy_(pca_tensor)
            self.WKey.weight.copy_(pca_tensor)
            self.WValue.weight.copy_(pca_tensor)
        print("Initialized WQuery/WKey/WValue from PCA of the input embeddings "
              "(no training, just SVD).")

    def forward(self, x):
        '''
        queries = inputs @ self.WQuery
        keys = inputs @ self.WKey
        values = inputs @ self.WValue
        '''

        print("=" * 70)
        print("STEP 1: Project the input into Query, Key, and Value spaces")
        print("=" * 70)
        print(f"Input x, shape {tuple(x.shape)}:\n{x}\n")

        # NOTE: using x here (the argument actually passed in), not the
        # global `inputs` -- using the global would silently ignore
        # whatever tensor the caller passes to this layer.
        queries = self.WQuery(x)
        keys = self.WKey(x)
        values = self.WValue(x)

        print(f"WQuery.weight, shape {tuple(self.WQuery.weight.shape)}:\n{self.WQuery.weight.data}\n")
        print(f"queries = WQuery(x), shape {tuple(queries.shape)}:\n{queries}\n")

        print(f"WKey.weight, shape {tuple(self.WKey.weight.shape)}:\n{self.WKey.weight.data}\n")
        print(f"keys = WKey(x), shape {tuple(keys.shape)}:\n{keys}\n")

        print(f"WValue.weight, shape {tuple(self.WValue.weight.shape)}:\n{self.WValue.weight.data}\n")
        print(f"values = WValue(x), shape {tuple(values.shape)}:\n{values}\n")

        if self.visualize:
            plot_matrix_multiplication(
                x, self.WQuery.weight.T, queries,
                A_label="Input (x)", B_label="WQuery.weight.T", C_label="Queries",
                A_row_labels=self.word_labels, C_row_labels=self.word_labels,
            )
            plot_matrix_multiplication(
                x, self.WKey.weight.T, keys,
                A_label="Input (x)", B_label="WKey.weight.T", C_label="Keys",
                A_row_labels=self.word_labels, C_row_labels=self.word_labels,
            )
            plot_matrix_multiplication(
                x, self.WValue.weight.T, values,
                A_label="Input (x)", B_label="WValue.weight.T", C_label="Values",
                A_row_labels=self.word_labels, C_row_labels=self.word_labels,
            )

        print("=" * 70)
        print("STEP 2: Attention scores = queries @ keys.T")
        print("=" * 70)
        attentionScores = queries @ keys.T
        print(f"attentionScores, shape {tuple(attentionScores.shape)} "
              f"(n_words x n_words -- every word's query compared against every word's key):")
        print(f"{attentionScores}\n")

        if self.visualize:
            plot_matrix_multiplication(
                queries, keys.T, attentionScores,
                A_label="Queries", B_label="Keys.T", C_label="Attention Scores",
                A_row_labels=self.word_labels, C_row_labels=self.word_labels,
            )

        print("=" * 70)
        print("STEP 3: Scale by sqrt(d_k), then softmax to get attention weights")
        print("=" * 70)
        d_k = keys.shape[1]
        print(f"d_k = keys.shape[1] = {d_k}, so we divide scores by sqrt({d_k}) = {d_k ** 0.5:.4f}")
        print("(Scaling prevents large dot products from making softmax overly peaked/unstable")
        print(" as d_k grows -- this is the same '/sqrt(d_k)' from the original Transformer paper.)\n")

        attentionWeights = torch.softmax(attentionScores / (d_k ** 0.5), dim=-1)
        print(f"attentionWeights, shape {tuple(attentionWeights.shape)}:")
        print(f"{attentionWeights}\n")
        print("Row sums (should all be 1.0, since softmax normalizes each word's "
              f"attention into a probability distribution): {attentionWeights.sum(dim=-1)}\n")

        print("=" * 70)
        print("STEP 4: Context vector = attentionWeights @ values")
        print("=" * 70)
        context_vector = attentionWeights @ values
        print(f"context_vector, shape {tuple(context_vector.shape)}:\n{context_vector}\n")

        if self.visualize:
            plot_matrix_multiplication(
                attentionWeights, values, context_vector,
                A_label="Attention Weights", B_label="Values", C_label="Context Vector",
                A_row_labels=self.word_labels, C_row_labels=self.word_labels,
            )

        # stash everything for inspection/visualization after forward() returns
        self.last_queries = queries
        self.last_keys = keys
        self.last_values = values
        self.last_attention_scores = attentionScores
        self.last_attention_weights = attentionWeights

        return context_vector


if __name__ == "__main__":
    args = parse_args()
    input_words = re.findall(r"\b[\w']+\b", args.sentence)

    print("Sentence Used:", args.sentence)
    print("Words:", input_words)

    # output_dim=6 here (instead of File 6's default of 3) gives PCA-init
    # and the Q/K/V projections more room to find real structure -- this
    # only affects this file; File 6's own default (3, for its 3D plots)
    # is untouched.
    inputs = generate_semantic_embeddings(input_words, output_dim=6)
    print("Embeddings (from word2vec, reduced to 6D):\n", inputs)

    d_in = inputs.shape[1]
    d_out = inputs.shape[1] - 1

    torch.manual_seed(15033019)
    selfAttention_v1 = SelfAttentionv1(d_in, d_out, visualize=True, word_labels=input_words)
    if args.pca_init:
        print("\nWQuery.weight BEFORE pca-init:\n", selfAttention_v1.WQuery.weight.data)
        selfAttention_v1.init_qkv_from_pca(inputs)
        print("WQuery.weight AFTER pca-init:\n", selfAttention_v1.WQuery.weight.data)
    context_vector = selfAttention_v1(inputs)

    print("=" * 70)
    print("FINAL: Context Vector")
    print("=" * 70)
    print(context_vector)

    # Visualize and quantify the resulting (trainable-weight) attention,
    # same toolkit used in the non-trainable-weights version, so you can
    # directly compare the two.
    plot_attention_heatmap(
        input_words, selfAttention_v1.last_attention_weights,
        title="Attention Weights Heatmap (with trainable Q/K/V)"
    )
    compute_attention_entropy(input_words, selfAttention_v1.last_attention_weights)
    print_top_k_attended(input_words, selfAttention_v1.last_attention_weights, k=3)
