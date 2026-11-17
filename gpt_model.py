# Ramakrishnan A
# File 12-B in the series on Building LLMs from scratch
# Full GPT Architecture: TransformerBlock + GPTModel + Greedy Next-Token
# Generation (`generate_text_simple`), assembling MultiHeadAttention (File 10-B)
# with LayerNorm and FeedForward (File 12-A).
# Lots of pranams to Sebastian Raschka

import argparse
import contextlib
import io
import tiktoken
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from multi_head_attention import MultiHeadAttention
from transformer_components import LayerNorm, FeedForward


GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size (GPT-2 BPE)
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of Transformer blocks
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
}

# Fast CPU-friendly preset for interactive inspection and training demos
GPT_CONFIG_MINI = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full GPTModel architecture (TransformerBlock + GPTModel + "
                    "greedy next-token generation) built from scratch."
    )
    parser.add_argument("--prompt", type=str, default="Every effort moves you",
                        help="Seed prompt for untrained greedy generation demo.")
    parser.add_argument("--max-new-tokens", type=int, default=6,
                        help="Number of new tokens to generate (default: 6).")
    parser.add_argument("--preset", choices=["mini", "124m"], default="mini",
                        help="Model size preset: 'mini' (fast CPU demo) or '124m' (GPT-2 Small).")
    return parser.parse_args()


class TransformerBlock(nn.Module):
    """
    A single Pre-LayerNorm GPT Transformer Block (Ch 4.5):
      x -> LayerNorm -> MultiHeadAttention -> Dropout -> (+ x)
        -> LayerNorm -> FeedForward (MLP)   -> Dropout -> (+ x)
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            visualize=False,
        )
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        # Suppress File 10-B's per-step educational prints during multi-layer runs
        with contextlib.redirect_stdout(io.StringIO()):
            x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    Complete GPT-2 style autoregressive language model (Ch 4.6):
      Token IDs -> Token Emb + Pos Emb -> Dropout -> N x TransformerBlock
                -> Final LayerNorm -> Linear Output Head (vocab_size logits)
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Autoregressive greedy decoding loop (Ch 4.7):
      At each step, crop to the last `context_size` tokens, run a forward
      pass, take the logits at the last token position, pick argmax, and
      append the predicted token ID to `idx`.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        # Focus only on the last time step: (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def plot_next_token_probs(tokenizer, logits_last_step, top_k=8,
                          figsize=(10, 4.5), save_path=None, show=True):
    """
    Plot the top-k predicted next-token probabilities from the final position's
    logits (before training, probabilities are near-uniform ~ 1/50257).
    """
    probs = torch.softmax(logits_last_step, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)
    labels = [repr(tokenizer.decode([tid.item()])) for tid in top_ids]
    vals = top_probs.detach().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(top_k)[::-1], vals, color="#4c6ef5")
    ax.set_yticks(range(top_k)[::-1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Probability (softmax)")
    ax.set_title(f"Top-{top_k} Next-Token Probabilities (Untrained GPTModel — Near Uniform Noise)")
    for bar, val in zip(bars, vals):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.5f}",
                va='center', ha='left', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    cfg = GPT_CONFIG_124M if args.preset == "124m" else GPT_CONFIG_MINI
    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)

    print("=" * 78)
    print("STEP 1: Verify a Single TransformerBlock Preserves Tensor Shape")
    print("=" * 78)
    sample_x = torch.rand(2, 6, cfg["emb_dim"])
    block = TransformerBlock(cfg)
    block_out = block(sample_x)
    print(f"Input shape : {tuple(sample_x.shape)}")
    print(f"Output shape: {tuple(block_out.shape)}  (exact same shape — blocks can be stacked N times!)\n")

    print("=" * 78)
    print(f"STEP 2: Instantiate Full GPTModel (preset={args.preset})")
    print("=" * 78)
    model = GPTModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    tied_params = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Config           : {cfg}")
    print(f"Total parameters : {total_params:,}")
    print(f"With weight tying: {tied_params:,} (when tok_emb and out_head share weights)\n")

    print("=" * 78)
    print("STEP 3: Forward Pass — Token IDs -> Logits Over Entire Vocabulary")
    print("=" * 78)
    encoded = tokenizer.encode(args.prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dim -> (1, seq_len)
    model.eval()
    with torch.no_grad():
        logits = model(encoded_tensor)
    print(f"Prompt           : {repr(args.prompt)}")
    print(f"Input IDs shape  : {tuple(encoded_tensor.shape)} -> {encoded}")
    print(f"Output logits    : {tuple(logits.shape)}  (batch=1, seq_len={len(encoded)}, vocab_size={cfg['vocab_size']:,})\n")

    print("=" * 78)
    print("STEP 4: Untrained Greedy Next-Token Generation (Motivating File 13 Training!)")
    print("=" * 78)
    out_ids = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
        context_size=cfg["context_length"],
    )
    decoded_text = tokenizer.decode(out_ids.squeeze(0).tolist())
    print(f"Generated IDs    : {out_ids.squeeze(0).tolist()}")
    print(f"Generated text   : {repr(decoded_text)}")
    print("\n(Because weights are still random, the output is gibberish — next up in")
    print(" File 13: training this GPTModel with AdamW & cross-entropy loss!)")

    plot_next_token_probs(tokenizer, logits[0, -1])
