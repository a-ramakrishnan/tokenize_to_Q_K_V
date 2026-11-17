# Ramakrishnan A
# File 11 in the series on Building LLMs from scratch
# End-to-End Input-to-Attention Bridge: connecting Stage 1-2 (tiktoken +
# createDataLoader + Token & Positional Embeddings) directly into Stage 7
# (MultiHeadAttention) on real text from book_of_wonders.txt.
# Lots of pranams to Sebastian Raschka

import argparse
import os
import tiktoken
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Reuse building blocks from earlier files in the series
from input_for_training import createDataLoader
from multi_head_attention import MultiHeadAttention
from multi_head_attention_wrapper import plot_multi_head_heatmaps


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end bridge from raw text -> BPE DataLoader -> "
                    "Token + Positional Embeddings -> MultiHeadAttention."
    )
    parser.add_argument("--text", type=str, default=os.path.join(SCRIPT_DIR, "book_of_wonders.txt"),
                        help="Path to the input .txt file (default: book_of_wonders.txt).")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for DataLoader (default: 2).")
    parser.add_argument("--max-length", type=int, default=8,
                        help="Context length (tokens per sequence, default: 8).")
    parser.add_argument("--stride", type=int, default=8,
                        help="Sliding-window stride (default: 8).")
    parser.add_argument("--emb-dim", type=int, default=16,
                        help="Embedding & MultiHeadAttention dimension (default: 16).")
    parser.add_argument("--num-heads", type=int, default=2,
                        help="Number of attention heads (must divide --emb-dim, default: 2).")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Attention dropout probability (default: 0.1).")
    return parser.parse_args()


def plot_pipeline_tensor_flow(token_labels, tok_emb, pos_emb, inp_emb, ctx_vec,
                              figsize=(18, 5.5), save_path=None, show=True):
    """
    Visualize how a single sequence (Batch 0) flows from:
      Token Embeddings  +  Positional Embeddings  =  Input Embeddings  ->  MHA Context Vectors
    """
    mats = [
        (tok_emb.detach().numpy(), "1. Token Embeddings\nnn.Embedding(50257, emb_dim)"),
        (pos_emb.detach().numpy(), "2. Positional Embeddings\nnn.Embedding(T, emb_dim)"),
        (inp_emb.detach().numpy(), "3. Input Embeddings\n(Token + Positional)"),
        (ctx_vec.detach().numpy(), "4. Multi-Head Context\n(Ready for Transformer Block)"),
    ]
    vmax = max(max(np.abs(m).max() for m, _ in mats), 1e-6)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 7, width_ratios=[3, 0.4, 3, 0.4, 3, 0.5, 3], figure=fig)

    positions = [0, 2, 4, 6]
    separators = [(1, "+"), (3, "="), (5, "→")]

    for idx, ((mat, title), col_idx) in enumerate(zip(mats, positions)):
        ax = fig.add_subplot(gs[0, col_idx])
        im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f"{title}\nshape {mat.shape}", fontsize=9.5)
        ax.set_xlabel("Embedding Dim")
        ax.set_yticks(range(len(token_labels)))
        if idx == 0:
            ax.set_yticklabels(token_labels, fontsize=8)
            ax.set_ylabel("Token (Batch 0)")
        else:
            ax.set_yticklabels([])

    for col_idx, sym in separators:
        ax_s = fig.add_subplot(gs[0, col_idx])
        ax_s.axis('off')
        ax_s.text(0.5, 0.5, sym, ha='center', va='center', fontsize=24, fontweight='bold')

    fig.colorbar(im, ax=fig.axes[-1], fraction=0.046, pad=0.04)
    plt.suptitle("End-to-End Tensor Flow: Token IDs → Embeddings → Multi-Head Attention Context",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()

    with open(args.text, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab  # 50,257 for GPT-2 BPE

    print("=" * 78)
    print("STEP 1: Load Text & Pull a Real Batch from createDataLoader (Files 2 & 3)")
    print("=" * 78)
    print(f"Text source : {os.path.basename(args.text)} ({len(raw_text):,} chars)")
    print(f"Tokenizer   : GPT-2 BPE (vocab_size = {vocab_size:,})")

    dataloader = createDataLoader(
        raw_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print(f"Batch input IDs shape : {tuple(inputs.shape)}  (batch_size={args.batch_size}, max_length={args.max_length})")
    print(f"Batch target IDs shape: {tuple(targets.shape)} (shifted by +1 token for next-token prediction)\n")

    for b_idx in range(inputs.shape[0]):
        in_ids = inputs[b_idx].tolist()
        tgt_ids = targets[b_idx].tolist()
        print(f"  Sample {b_idx} Input  IDs   : {in_ids}")
        print(f"  Sample {b_idx} Input  Text  : {repr(tokenizer.decode(in_ids))}")
        print(f"  Sample {b_idx} Target IDs   : {tgt_ids}")
        print(f"  Sample {b_idx} Target Text  : {repr(tokenizer.decode(tgt_ids))}\n")

    print("=" * 78)
    print("STEP 2: Token Embeddings + Positional Embeddings (File 5-A)")
    print("=" * 78)
    torch.manual_seed(123)
    token_embedding_layer = nn.Embedding(vocab_size, args.emb_dim)
    pos_embedding_layer = nn.Embedding(args.max_length, args.emb_dim)

    token_embeddings = token_embedding_layer(inputs)
    pos_indices = torch.arange(args.max_length)
    pos_embeddings = pos_embedding_layer(pos_indices)
    input_embeddings = token_embeddings + pos_embeddings

    print(f"token_embeddings shape : {tuple(token_embeddings.shape)}")
    print(f"pos_embeddings shape   : {tuple(pos_embeddings.shape)}  (broadcast across batch)")
    print(f"input_embeddings shape : {tuple(input_embeddings.shape)}\n")

    print("=" * 78)
    print("STEP 3: Pass Input Embeddings into MultiHeadAttention (File 10-B)")
    print("=" * 78)
    mha = MultiHeadAttention(
        d_in=args.emb_dim,
        d_out=args.emb_dim,
        context_length=args.max_length,
        dropout=args.dropout,
        num_heads=args.num_heads,
        qkv_bias=False,
        visualize=False,
    )
    mha.eval()  # clean deterministic attention weights for inspection
    with torch.no_grad():
        context_vectors = mha(input_embeddings)

    print("=" * 78)
    print("FINAL SUMMARY: Ready for Transformer Block (LayerNorm + MLP + Residual)")
    print("=" * 78)
    print(f"Input token IDs       : {tuple(inputs.shape)}")
    print(f"Combined embeddings   : {tuple(input_embeddings.shape)}")
    print(f"MHA context vectors   : {tuple(context_vectors.shape)}")
    print(f"Next-token target IDs : {tuple(targets.shape)}")

    # Decode individual BPE tokens in Batch 0 for clean axis labels
    batch0_tokens = [repr(tokenizer.decode([tid])).strip("'") for tid in inputs[0].tolist()]
    plot_pipeline_tensor_flow(
        batch0_tokens,
        token_embeddings[0],
        pos_embeddings,
        input_embeddings[0],
        context_vectors[0],
    )
    head_weights_batch0 = [mha.last_attention_weights[0, h] for h in range(args.num_heads)]
    plot_multi_head_heatmaps(batch0_tokens, head_weights_batch0)
