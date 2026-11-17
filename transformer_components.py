# Ramakrishnan A
# File 12-A in the series on Building LLMs from scratch
# Transformer Building Blocks: LayerNorm, GELU Activation, and FeedForward (MLP)
# — the three non-attention components inside every GPT Transformer block.
# Lots of pranams to Sebastian Raschka

import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step-by-step walkthrough and visualization of LayerNorm, "
                    "GELU, FeedForward (MLP), and Residual Connections."
    )
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for sample tensors (default: 2).")
    parser.add_argument("--seq-len", type=int, default=5,
                        help="Sequence length (number of tokens, default: 5).")
    parser.add_argument("--emb-dim", type=int, default=8,
                        help="Embedding dimension (default: 8).")
    return parser.parse_args()


class LayerNorm(nn.Module):
    """
    Layer Normalization (matching Sebastian Raschka Ch 4.2 / GPT-2).

    Why we need LayerNorm:
      Activations flowing through many Transformer layers can drift in scale
      (exploding or vanishing). LayerNorm normalizes EACH token's feature vector
      independently across `dim=-1` to mean = 0 and variance = 1, then applies
      learnable per-feature `scale` (gamma) and `shift` (beta) parameters.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # GPT-2 uses biased variance (unbiased=False, dividing by N instead of N-1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GPT-2 tanh approximation, Ch 4.3).

    Unlike ReLU(x) = max(0, x), which has a sharp corner at 0 and zero gradient
    for all x < 0 ("dead neurons"), GELU smoothly gates values by how likely
    they are under a standard Gaussian, allowing small negative values and
    clean gradients everywhere.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network / MLP (Ch 4.3).

    Expands each token's representation from `emb_dim` to `4 * emb_dim` so the
    model has a richer non-linear workspace to transform features learned during
    self-attention, applies GELU, and projects back to `emb_dim`.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


def plot_layernorm_effect(x_before, x_after, figsize=(13, 5), save_path=None, show=True):
    """
    Visualize Batch 0 embeddings before and after LayerNorm, alongside each
    token's mean and variance.
    """
    b_np = x_before[0].detach().numpy()
    a_np = x_after[0].detach().numpy()
    vmax = max(np.abs(b_np).max(), np.abs(a_np).max(), 1e-6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for ax, mat, title in [
        (ax1, b_np, f"Before LayerNorm (Batch 0)\nMean={b_np.mean():.2f}, Std={b_np.std():.2f}"),
        (ax2, a_np, f"After LayerNorm (Batch 0)\nPer-row Mean≈0.00, Var≈1.00"),
    ]:
        im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Token Position")
        ax.set_yticks(range(mat.shape[0]))
        if mat.shape[0] * mat.shape[1] <= 80:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    col = 'white' if abs(mat[i, j]) > vmax * 0.6 else 'black'
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center', fontsize=7, color=col)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("LayerNorm Centers & Scales Each Token Vector Independently",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_gelu_vs_relu(figsize=(10, 4.5), save_path=None, show=True):
    """
    Plot GELU vs. ReLU over [-3, 3] to highlight smoothness at x=0 and
    non-zero negative lobe.
    """
    gelu, relu = GELU(), nn.ReLU()
    x = torch.linspace(-3, 3, 200)
    y_gelu, y_relu = gelu(x), relu(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(x.numpy(), y_gelu.numpy(), label="GELU", linewidth=2.2, color="#1f77b4")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("GELU Activation (Smooth, Small Negative Dip)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("GELU(x)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x.numpy(), y_relu.numpy(), label="ReLU", linewidth=2.2, color="#d62728")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_title("ReLU Activation (Sharp Corner at 0, Dead for x < 0)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("ReLU(x)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_feedforward_expansion(x_in, hidden, x_out, figsize=(16, 4.8), save_path=None, show=True):
    """
    Visualize the 4x bottleneck expansion inside FeedForward:
      Input (T x emb_dim) -> GELU(Linear(x)) (T x 4*emb_dim) -> Output (T x emb_dim)
    """
    A = x_in[0].detach().numpy()
    B = hidden[0].detach().numpy()
    C = x_out[0].detach().numpy()
    vmax = max(np.abs(A).max(), np.abs(B).max(), np.abs(C).max(), 1e-6)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 5, width_ratios=[1.5, 0.35, 4.0, 0.35, 1.5], figure=fig)

    for col_idx, mat, title in [
        (0, A, f"Input x\nshape {A.shape}"),
        (2, B, f"Hidden (4× Expansion + GELU)\nshape {B.shape}"),
        (4, C, f"FeedForward Output\nshape {C.shape}"),
    ]:
        ax = fig.add_subplot(gs[0, col_idx])
        im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Feature Dim")
        ax.set_yticks(range(mat.shape[0]))
        if col_idx == 0:
            ax.set_ylabel("Token Position")
        else:
            ax.set_yticklabels([])

    for sep_idx in [1, 3]:
        ax_s = fig.add_subplot(gs[0, sep_idx])
        ax_s.axis('off')
        ax_s.text(0.5, 0.5, "→", ha='center', va='center', fontsize=24, fontweight='bold')

    fig.colorbar(im, ax=fig.axes[-1], fraction=0.046, pad=0.04)
    plt.suptitle("FeedForward (MLP): Expand 4× Into Richer Feature Space, Then Project Back",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


class ExampleDeepNeuralNetwork(nn.Module):
    """
    5-layer toy network to demonstrate why residual (shortcut) connections
    prevent vanishing gradients in deep Transformers (Ch 4.4).
    """
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.0]])
    loss = nn.MSELoss()(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"  {name:<18} mean |grad| = {param.grad.abs().mean().item():.6f}")


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(123)

    # Create a skewed sample batch so LayerNorm's effect is obvious
    x = torch.randn(args.batch_size, args.seq_len, args.emb_dim) * 3.5 + 2.0

    print("=" * 78)
    print("STEP 1: LayerNorm (Mean = 0, Variance = 1 per token)")
    print("=" * 78)
    print(f"Input x shape: {tuple(x.shape)}")
    print(f"Before LayerNorm — token means (Batch 0): {x[0].mean(dim=-1)}")
    print(f"Before LayerNorm — token vars  (Batch 0): {x[0].var(dim=-1, unbiased=False)}\n")

    ln = LayerNorm(args.emb_dim)
    out_ln = ln(x)
    print(f"After LayerNorm  — token means (Batch 0): {out_ln[0].mean(dim=-1)}")
    print(f"After LayerNorm  — token vars  (Batch 0): {out_ln[0].var(dim=-1, unbiased=False)}")

    # Verify against PyTorch's built-in nn.LayerNorm
    pytorch_ln = nn.LayerNorm(args.emb_dim)
    max_diff = torch.max(torch.abs(out_ln - pytorch_ln(x))).item()
    print(f"Max difference vs. torch.nn.LayerNorm: {max_diff:.2e}\n")
    plot_layernorm_effect(x, out_ln)

    print("=" * 78)
    print("STEP 2: GELU Activation vs. ReLU")
    print("=" * 78)
    gelu = GELU()
    sample_vals = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    print(f"Inputs    : {sample_vals.tolist()}")
    print(f"ReLU(x)   : {[round(v, 4) for v in torch.relu(sample_vals).tolist()]}")
    print(f"GELU(x)   : {[round(v, 4) for v in gelu(sample_vals).tolist()]}\n")
    plot_gelu_vs_relu()

    print("=" * 78)
    print("STEP 3: FeedForward (MLP) 4x Expansion & Contraction")
    print("=" * 78)
    ffn = FeedForward(args.emb_dim)
    with torch.no_grad():
        hidden = ffn.layers[1](ffn.layers[0](out_ln))
        out_ffn = ffn(out_ln)
    print(f"Input shape  : {tuple(out_ln.shape)}")
    print(f"Hidden shape : {tuple(hidden.shape)}  (expanded 4× from {args.emb_dim} to {4 * args.emb_dim})")
    print(f"Output shape : {tuple(out_ffn.shape)}  (projected back to {args.emb_dim})\n")
    plot_feedforward_expansion(out_ln, hidden, out_ffn)

    print("=" * 78)
    print("STEP 4: Residual (Shortcut) Connections Prevent Vanishing Gradients")
    print("=" * 78)
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print("Gradients WITHOUT shortcut connections (notice layer 0 shrinks):")
    print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print("\nGradients WITH shortcut connections (healthy gradient flow to layer 0):")
    print_gradients(model_with_shortcut, sample_input)
