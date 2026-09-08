# Ramakrishnan A
# File 6 in the series on Building LLMs from scratch
# Lots of pranams to Sebastian Raschka

# Starting Attention Mechanism

import torch
import argparse
import re
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.widgets import CheckButtons
import sys

# Anchor the cached word2vec file to this script's own directory, not the
# current working directory -- so it's found consistently no matter where
# you run the script from, and only ever downloaded once.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORD2VEC_LOCAL_PATH = os.path.join(SCRIPT_DIR, "word2vec-google-news-300.kv")

# print("Sentence Used \n")
# print("______________\n")
# print("Cats scare mice while mice terrify elephants.")
# print("______________\n")

# input_words = ["Cats", "scare", 'mice', 'while', 'mice', 'terrify', 'elephants']

# inputs = torch.tensor([
#     [0.45, 0.58, 0.66],
#     [0.77, 0.11, 0.12],
#     [0.4, 0.62, 0.70],
#     [0.05, 0.15, 0.72],
#     [0.4, 0.62, 0.70],
#     [0.80, 0.12, 0.2],
#     [0.65, 0.78, 0.83]
# ])
#
OUTPUT_DIM = 3

def generate_semantic_embeddings(input_words, model=None, local_path=WORD2VEC_LOCAL_PATH):
    """
    Generate embeddings where semantically similar words end up close
    together, using a pretrained word2vec model rather than random noise.
    Always returns exactly 3 dimensions, to plug directly into
    plotting_fn / plot_original_vs_context.

    input_words : list of words
    model       : an already-loaded gensim KeyedVectors model (optional --
                  pass this in if calling repeatedly, to avoid reloading
                  the ~1.6GB model each time)
    local_path  : path to a locally saved KeyedVectors file. If it exists,
                  loads from disk (fast, no network). If not, downloads
                  once via gensim's downloader and saves it here for next
                  time -- so the ~1.6GB download only ever happens once.
                  Defaults to a path next to this script file, so it's
                  found consistently regardless of the working directory.

    Returns a torch.Tensor of shape (len(input_words), 3), scaled to
    [0, 1] per dimension to match the existing plot axis limits.
    """
    if model is None:
        if os.path.exists(local_path):
            from gensim.models import KeyedVectors
            print(f"Loading word2vec model from local cache: {local_path}")
            model = KeyedVectors.load(local_path)
        else:
            import gensim.downloader as api
            print("Local cache not found. Downloading word2vec-google-news-300 "
                  "(one-time, ~1.6GB)...")
            model = api.load("word2vec-google-news-300")
            print(f"Saving to {local_path} so future runs skip the download...")
            model.save(local_path)

    vectors = []
    missing = []
    for word in input_words:
        key = word.lower()
        if key in model:
            vectors.append(model[key])
        else:
            missing.append(word)
            vectors.append(None)

    if missing:
        print(f"Warning: {missing} not found in the pretrained vocabulary; "
              f"using the average of the other words' vectors as a fallback.")
        known = np.array([v for v in vectors if v is not None])
        fallback = known.mean(axis=0) if len(known) > 0 else np.zeros(model.vector_size)
        vectors = [v if v is not None else fallback for v in vectors]

    vectors = np.array(vectors, dtype=float)
    n_words = vectors.shape[0]

    # PCA via SVD. The number of usable components is capped by
    # min(n_words - 1, vector_size) -- with very few words, that can be
    # LESS than 3, so we explicitly pad with zero columns afterward to
    # guarantee exactly 3 output dimensions no matter what.
    centered = vectors - vectors.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(OUTPUT_DIM, Vt.shape[0])
    reduced = centered @ Vt[:n_components].T

    if n_components < OUTPUT_DIM:
        pad_width = OUTPUT_DIM - n_components
        reduced = np.pad(reduced, ((0, 0), (0, pad_width)), mode='constant')

    # Scale into [0, 1] per dimension, matching the plots' axis limits.
    # Guard against a zero range (e.g. a padded all-zero column, or a
    # single repeated word) so we don't divide by zero.
    mins = reduced.min(axis=0)
    maxs = reduced.max(axis=0)
    ranges = np.where(maxs - mins > 1e-8, maxs - mins, 1.0)
    scaled = (reduced - mins) / ranges

    assert scaled.shape == (n_words, OUTPUT_DIM), (
        f"Expected shape ({n_words}, {OUTPUT_DIM}), got {scaled.shape}"
    )

    return torch.tensor(scaled, dtype=torch.float32)

def checkpoint(label=""):
    """
    Pause execution and wait for the user to press Enter to continue.
    Type 'q' (then Enter) to quit the script early instead.
    """
    prompt = f"\n--- {label} --- Press Enter to continue, or 'q' to quit: " if label else "\nPress Enter to continue, or 'q' to quit: "
    response = input(prompt)
    if response.strip().lower() == 'q':
        print("Stopped by user.")
        sys.exit(0)

def generate_random_embeddings(input_words, embedding_dim=3, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(len(input_words), embedding_dim)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate random embeddings for a sentence's words.")
    parser.add_argument("sentence", type=str, help="The sentence to tokenize into words")
    parser.add_argument("--embedding-dim", type=int, default=3, help="Size of each embedding vector (default: 3)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    return parser.parse_args()


def plot_original_vs_context(input_words, original_inputs, context_vector, figsize=(12, 10)):
    assert len(input_words) == len(original_inputs) == len(context_vector), (
        f"Length mismatch: {len(input_words)} words, {len(original_inputs)} original "
        f"vectors, {len(context_vector)} context vectors"
    )
    orig_np = original_inputs.detach().numpy()
    ctx_np = context_vector.detach().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.28)

    artists_by_key = {}
    for i, word in enumerate(input_words):
        ox, oy, oz = orig_np[i]
        cx, cy, cz = ctx_np[i]

        orig_pt = ax.scatter(ox, oy, oz, color='tab:blue', s=70, marker='o')
        ctx_pt = ax.scatter(cx, cy, cz, color='tab:orange', s=70, marker='^')
        orig_txt = ax.text(ox, oy, oz, word, fontsize=9, color='tab:blue')
        ctx_txt = ax.text(cx, cy, cz, f"{word} (ctx)", fontsize=9, color='tab:orange')

        # vectors from the origin, same style as your original plotting_fn
        orig_line = ax.quiver(0, 0, 0, ox, oy, oz, color='tab:blue', alpha=0.5,
                               arrow_length_ratio=0.05, linewidth=1)
        ctx_line = ax.quiver(0, 0, 0, cx, cy, cz, color='tab:orange', alpha=0.5,
                              arrow_length_ratio=0.05, linewidth=1)

        # dashed line showing the shift from original -> context vector
        shift_line, = ax.plot([ox, cx], [oy, cy], [oz, cz], color='gray', linestyle='--', linewidth=1)

        key = f"{word} #{i}"
        artists_by_key[key] = [orig_pt, ctx_pt, orig_txt, ctx_txt, orig_line, ctx_line, shift_line]

    # Mark the mean of the original inputs -- this is what the context
    # vectors are clustering around
    mean_np = orig_np.mean(axis=0)
    mean_pt = ax.scatter(*mean_np, color='black', s=200, marker='*', zorder=5)
    mean_txt = ax.text(*mean_np, "  mean(inputs)", fontsize=10, color='black', fontweight='bold')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
    plt.title('Original Embeddings vs Context Vectors\n(context vectors cluster near the mean)')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=9, label='Original'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='tab:orange', markersize=9, label='Context Vector'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=13, label='Mean of Inputs'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    check_ax = plt.axes([0.02, 0.3, 0.2, 0.5])
    labels = list(artists_by_key.keys())
    check = CheckButtons(check_ax, labels, [True] * len(labels))

    def toggle(label):
        for artist in artists_by_key[label]:
            artist.set_visible(not artist.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(toggle)
    plt.show()
    return check

def plotting_fn(input_words, inputs, figsize=(12, 10), save_path=None, show=True):
    """
    Plot each word's embedding as a point + a vector from the origin,
    with per-word checkboxes to toggle visibility.

    input_words : list of words (length n)
    inputs      : (n, 3) tensor of embeddings
    """
    assert len(input_words) == len(inputs), (
        f"Length mismatch: {len(input_words)} words, {len(inputs)} embedding vectors"
    )

    coords = inputs.detach().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.28)

    artists_by_key = {}
    for i, word in enumerate(input_words):
        x, y, z = coords[i]

        pt = ax.scatter(x, y, z, color='tab:blue', s=70, marker='o')
        txt = ax.text(x, y, z, word, fontsize=10, color='tab:blue')
        line = ax.quiver(0, 0, 0, x, y, z, color='tab:blue', alpha=0.6,
                          arrow_length_ratio=0.05, linewidth=1)

        key = f"{word} #{i}"   # unique key -- plain word collides on repeats (e.g. "mice")
        artists_by_key[key] = [pt, txt, line]

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
    plt.title('3D Plot - Word Embeddings (vectors from origin)')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=9, label='Embedding'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    check_ax = plt.axes([0.02, 0.3, 0.2, 0.5])
    labels = list(artists_by_key.keys())
    check = CheckButtons(check_ax, labels, [True] * len(labels))

    def toggle(label):
        for artist in artists_by_key[label]:
            artist.set_visible(not artist.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(toggle)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return check

if __name__ == "__main__":
    args = parse_args()
    input_words = re.findall(r"\b[\w']+\b", args.sentence)

    print("Sentence Used:", args.sentence)
    print("Words:", input_words)

    # inputs = generate_random_embeddings(input_words, embedding_dim=args.embedding_dim, seed=args.seed)
    inputs = generate_semantic_embeddings(input_words)
    print("Embeddings:\n", inputs)
    plotting_fn(input_words, inputs)

    checkpoint("Good with Embeddings")

    # Attention Scores
    # Dot Product formula
    print("Let's first get the magnitudes")
    magnitudes = torch.norm(inputs, dim = 1)
    for input_word, magnitude in zip(input_words, magnitudes):
        print(f"{input_word}: {magnitude.item():.4f}")

    print(f"Let's convert it for 1 word/token-> {input_words[1]}: {inputs[1]}")
    query = inputs[1]
    attn_scores_1 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_1[i] = torch.dot(x_i, query)
    print(f"Attention score of {input_words[1]} is ", attn_scores_1)
    checkpoint("Attention Score of a single word and all words completed")

    print("Now to Attention Weights\n")
    print("Normalizing it with a Summation")
    attention_weights_1_simple = attn_scores_1 / attn_scores_1.sum()
    print("Attention Weights: ", attention_weights_1_simple)
    print("Sum of Attention Weights", attention_weights_1_simple.sum())
    checkpoint("Good with Attention Weights - For 1 word/token - Simple")

    print("\nNow with Softmax")
    attention_scores_1_softmax = tf.convert_to_tensor(attn_scores_1.numpy())
    attention_weights_1_softmax = tf.nn.softmax(attention_scores_1_softmax)
    print("Attention Weights: ", attention_weights_1_softmax)
    print("Sum of Attention Weights", attention_weights_1_softmax.numpy().sum())
    checkpoint("Good with Attention Weights - For 1 word/token - Softmax")

    print("\nComplete Attention Scores")
    attention_scores = inputs @ inputs.T
    print("Shape of Attention Scores Matrix", attention_scores.shape)
    print("Attention Score Matrix\n", attention_scores)
    checkpoint("Good with Attention Scores - Simple")

    print("Moving to Softmax")
    attention_weights = torch.softmax(attention_scores, dim=1)
    print("Shape of Attention Weights Matrix", attention_weights.shape)
    print("Attention Weights Matrix\n", attention_weights)
    print("Sum of the rows independently -> ", attention_weights.sum(dim=1))
    checkpoint("Good with Attention Weights - Softmax")

    print("\nNow onto the Context Vectors")
    context_vector = attention_weights @ inputs
    print("Context Vector \n", context_vector)
    print(f"Context vector of {input_words[1]} is ", context_vector[1])
    checkpoint("Good with Context Vector - Single word")

    print("Append Context Vector to Inputs")
    original_inputs = inputs.clone()

    print("Append Context Vector to Inputs")
    inputs = torch.cat((inputs, context_vector), dim=0)
    print("Updated Input Tensor\n", inputs)
    print("Updated Words list\n", input_words)
    checkpoint("Good with Context Vectors")

    print("\nComparing Original Embeddings vs Context Vectors for every word")
    print(f"{'Word':<12}{'Original Embedding':<28}{'Context Vector':<28}")
    print("-" * 68)
    for word, orig, ctx in zip(input_words, original_inputs, context_vector):
        orig_str = "[" + ", ".join(f"{v.item():.4f}" for v in orig) + "]"
        ctx_str = "[" + ", ".join(f"{v.item():.4f}" for v in ctx) + "]"
        print(f"{word:<12}{orig_str:<28}{ctx_str:<28}")

    # Showing WHY context vectors cluster together: they all land close to
    # the mean of the original inputs, since the attention weights here are
    # close to uniform (no trainable weights yet to sharpen them).
    mean_of_inputs = original_inputs.mean(dim=0)
    print("\nMean of original inputs:", mean_of_inputs)
    print(f"\n{'Word':<12}{'Context Vector':<28}{'Distance from Mean':<20}")
    print("-" * 60)
    for word, ctx in zip(input_words, context_vector):
        dist = torch.norm(ctx - mean_of_inputs).item()
        ctx_str = "[" + ", ".join(f"{v.item():.4f}" for v in ctx) + "]"
        print(f"{word:<12}{ctx_str:<28}{dist:<20.4f}")

    print(f"\nSpread (std) of ORIGINAL inputs: {original_inputs.std().item():.4f}")
    print(f"Spread (std) of CONTEXT vectors:  {context_vector.std().item():.4f}")
    print(f"Context vectors are {(original_inputs.std()/context_vector.std()).item():.1f}x tighter than the originals")

    plot_original_vs_context(input_words, original_inputs, context_vector)
