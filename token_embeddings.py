# Ramakrishnan A
# File 5-A in the series on Building LLMs from scratch
# Lots of pranams to Sebastian Raschka

# Embedding layer Weight Matrix
# Token IDs are rows
# Dimensions/Features are columns

# Embedding Layer Weight Matrix is optimized using Backpropogation

# For now, we will only create the embedding layer matrix

import torch
import tiktoken
from transformers import GPT2Model
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use("Agg")

def visualize_linear_layer(layer, x, save_path=None, show=True):
    """
    layer : a torch.nn.Linear instance
    x     : a 1D tensor/list of length layer.in_features
    """
    W = layer.weight.detach().tolist()   # shape (out_features, in_features)
    b = layer.bias.detach().tolist()     # shape (out_features,)
    x = [float(v) for v in x]

    out_features = len(W)
    in_features = len(W[0])
    assert len(x) == in_features, f"x has length {len(x)}, expected {in_features}"

    # ---- 1. print the computation ----
    print(f"Linear({in_features}, {out_features}) -- weight shape ({out_features}, {in_features}), "
          f"bias shape ({out_features},)\n")

    y = []
    for i in range(out_features):
        print(f"y{i+1}:")
        total = b[i]
        for j in range(in_features):
            w = W[i][j]
            contribution = w * x[j]
            total += contribution
            print(f"  w{i+1}_{j+1} (x{j+1}={x[j]:.3f} -> y{i+1}) = {w: .4f}  "
                  f"(contributes {contribution: .4f})")
        print(f"  bias{i+1} = {b[i]: .4f}")
        print(f"  y{i+1} = {total: .4f}\n")
        y.append(total)

    print("Input x  =", [round(v, 4) for v in x])
    print("Output y =", [round(v, 4) for v in y])

    # ---- 2. draw the diagram ----
    fig_height = max(4, 0.55 * max(in_features, out_features))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    input_y = _spread(in_features)
    output_y = _spread(out_features)
    input_x, output_x = 0.15, 0.85

    colors = plt.get_cmap("tab10").colors
    out_colors = [colors[i % len(colors)] for i in range(out_features)]

    # draw connections, colored by which output neuron they feed
    for i in range(out_features):
        for j in range(in_features):
            ax.plot([input_x, output_x], [input_y[j], output_y[i]],
                     color=out_colors[i], alpha=0.55, linewidth=1.2, zorder=1)

    # input nodes, labeled with the actual x values provided
    for j in range(in_features):
        ax.scatter([input_x], [input_y[j]], s=900, facecolor="#EEEDFE",
                    edgecolor="#534AB7", linewidth=1.5, zorder=2)
        ax.text(input_x, input_y[j], f"{x[j]:.2g}", ha="center", va="center",
                 fontsize=11, zorder=3)
        ax.text(input_x - 0.08, input_y[j], f"x{j+1}", ha="right", va="center", fontsize=9)

    # output nodes, colored to match their incoming lines, labeled with computed y
    for i in range(out_features):
        ax.scatter([output_x], [output_y[i]], s=1100, facecolor="white",
                    edgecolor=out_colors[i], linewidth=2.2, zorder=2)
        ax.text(output_x, output_y[i], f"y{i+1}", ha="center", va="center",
                 fontsize=11, zorder=3)
        ax.text(output_x + 0.1, output_y[i], f"= {y[i]:.3f}", ha="left", va="center",
                 fontsize=9, color=out_colors[i])

    ax.set_xlim(-0.1, 1.3)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")

    if in_features <= 8:
        title = f"Linear({in_features}, {out_features})  --  x = {[round(v, 2) for v in x]}"
    else:
        nonzero = [(j, v) for j, v in enumerate(x) if v != 0]
        if len(nonzero) <= 3:
            active = ", ".join(f"x{j+1}={v:.2g}" for j, v in nonzero)
            title = f"Linear({in_features}, {out_features})  --  {active}"
        else:
            title = f"Linear({in_features}, {out_features})  --  {len(nonzero)} nonzero inputs"
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved diagram to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return y


def _spread(n):
    """Evenly spaced y-positions for n nodes, centered in [0, 1]."""
    if n == 1:
        return [0.5]
    return [1 - i / (n - 1) for i in range(n)]


# Load the tokenizer
# Using the token set of GPT2
# We want to use the token IDs from this tokenizer and not the weights
tokenizer = tiktoken.get_encoding("gpt2")
ids = tokenizer.encode("The luggage that the exhausted tourists forgot to retrieve from the carousel after their flight landed in Rome was lost.")
print(ids)
print(tokenizer.decode(ids))
print("Length of input Ids", len(ids))

embedding_training_dimensions = 3
vocab_size = len(ids)

input_ids = torch.tensor(ids)
torch.manual_seed(15033019)

embedding_layer = torch.nn.Embedding(vocab_size, embedding_training_dimensions)

print("\nEmbedding Layer Weights")
print(embedding_layer.weight)
print(embedding_layer.weight.shape)
print(embedding_layer.num_embeddings)
print(embedding_layer.embedding_dim)

print(embedding_layer(torch.tensor([2])))
print(embedding_layer(torch.tensor([10])))

print("Let us see some actual embeddings from the GPT2 Model")
model = GPT2Model.from_pretrained("gpt2")
token_id = ids[1]
print(f"ids[1] is {ids[1]} and word is {tokenizer.decode([ids[1]])}")
vector = model.wte.weight[token_id]
print(vector.shape)
print("\n Actual Embeddings")
print(vector)

linear_weights = torch.nn.Linear(vocab_size, embedding_training_dimensions)
print("Linear Layer", linear_weights.weight)
print(linear_weights.weight.shape)
W = linear_weights.weight        # shape (3, 22)
b = linear_weights.bias          # shape (3,)

out_features, in_features = W.shape

print(f"Weight matrix W, shape ({out_features}, {in_features}):\n")

header = "        " + "".join(f"{'x'+str(j+1):>8}" for j in range(in_features))
print(header)

for i in range(out_features):
    row_values = "".join(f"{W[i, j].item():8.3f}" for j in range(in_features))
    print(f"y{i+1}  [{row_values} ]")

print(f"\nBias vector b, shape ({out_features},):")
print("[" + "".join(f"{b[i].item():8.3f}" for i in range(out_features)) + " ]")


with torch.no_grad():
    linear_weights.weight.copy_(embedding_layer.weight.T)
    # (3, vocab_size) = (vocab_size, 3).T
# one-hot vector for whichever token you want to visualize
token_id = 5   # <- change this to any token id in [0, vocab_size)
x = torch.zeros(vocab_size)
x[token_id] = 1.0

y = visualize_linear_layer(linear_weights, x, save_path="embedding_layer.png")
y_minus_bias = torch.tensor(y) - linear_weights.bias.detach()
print("y - bias                     =", y_minus_bias)
print("embedding_layer.weight[token_id] =", embedding_layer.weight[token_id])
print("Match (within float tolerance)?", torch.allclose(y_minus_bias, embedding_layer.weight[token_id], atol=1e-4))
