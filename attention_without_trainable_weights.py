# Ramakrishnan A
# File 6 in the series on Building LLMs from scratch
# Lots of pranams to Sebastian Raschka

# Starting Attention Mechanism

import torch
import argparse
import re
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.widgets import CheckButtons

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

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
    plt.title('Original Embeddings vs Context Vectors (vectors from origin)')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=9, label='Original'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='tab:orange', markersize=9, label='Context Vector'),
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

def plotting_fn(input_words, inputs):
    x_cood = inputs[:,0].numpy()
    y_cood = inputs[:,1].numpy()
    z_cood = inputs[:,2].numpy()

    # Create 3D plot
    embeddings_figure = plt.figure(figsize=(12, 10))
    ax = embeddings_figure.add_subplot(111, projection='3d')

    for x,y,z, input_word in zip(x_cood, y_cood, z_cood, input_words):
        ax.scatter(x,y,z)
        ax.text(x,y,z, input_word, fontsize=10)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    colours = ['r', 'g','b','c','m','y', 'purple']
    for (x,y,z, input_word, colours) in zip(x_cood, y_cood, z_cood, input_words, colours):
        ax.quiver(0,0,0,x,y,z, color = colours, arrow_length_ratio=0.03)
        ax.text(x,y,z, input_word, fontsize=10, color = colours)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])

    plt.title('3D Plot - With my own values')
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    input_words = re.findall(r"\b[\w']+\b", args.sentence)

    print("Sentence Used:", args.sentence)
    print("Words:", input_words)

    inputs = generate_random_embeddings(input_words, embedding_dim=args.embedding_dim, seed=args.seed)
    print("Embeddings:\n", inputs)
    plotting_fn(input_words, inputs)

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

    print("Now to Attention Weights\n")
    print("Normalizing it with a Summation")
    attention_weights_1_simple = attn_scores_1 / attn_scores_1.sum()
    print("Attention Weights: ", attention_weights_1_simple)
    print("Sum of Attention Weights", attention_weights_1_simple.sum())

    print("\nNow with Softmax")
    attention_scores_1_softmax = tf.convert_to_tensor(attn_scores_1.numpy())
    attention_weights_1_softmax = tf.nn.softmax(attention_scores_1_softmax)
    print("Attention Weights: ", attention_weights_1_softmax)
    print("Sum of Attention Weights", attention_weights_1_softmax.numpy().sum())

    print("\nComplete Attention Scores")
    attention_scores = inputs @ inputs.T
    print("Shape of Attention Scores Matrix", attention_scores.shape)
    print("Attention Score Matrix\n", attention_scores)

    print("Moving to Softmax")
    attention_weights = torch.softmax(attention_scores, dim=1)
    print("Shape of Attention Weights Matrix", attention_weights.shape)
    print("Attention Weights Matrix\n", attention_weights)
    print("Sum of the rows independently -> ", attention_weights.sum(dim=1))

    print("\nNow onto the Context Vectors")
    context_vector = attention_weights @ inputs
    print("Context Vector \n", context_vector)
    print(f"Context vector of {input_words[1]} is ", context_vector[1])

    print("Append Context Vector to Inputs")
    original_inputs = inputs.clone()

    print("Append Context Vector to Inputs")
    inputs = torch.cat((inputs, context_vector), dim=0)
    print("Updated Input Tensor\n", inputs)
    print("Updated Words list\n", input_words)

    print("\nComparing Original Embeddings vs Context Vectors for every word")
    print(f"{'Word':<12}{'Original Embedding':<28}{'Context Vector':<28}")
    print("-" * 68)
    for word, orig, ctx in zip(input_words, original_inputs, context_vector):
        orig_str = "[" + ", ".join(f"{v.item():.4f}" for v in orig) + "]"
        ctx_str = "[" + ", ".join(f"{v.item():.4f}" for v in ctx) + "]"
        print(f"{word:<12}{orig_str:<28}{ctx_str:<28}")

    plot_original_vs_context(input_words, original_inputs, context_vector)
