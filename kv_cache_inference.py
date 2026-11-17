# Ramakrishnan A
# File 14 in the series on Building LLMs from scratch
# KV Cache Inference Walkthrough: visualizing how Key (K) and Value (V)
# caches are built during the Prefill phase and grown 1 token at a time
# during the Decode phase, with an exact equivalence check and speedup benchmark.
# Lots of pranams to Sebastian Raschka

import argparse
import time
import tiktoken
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformer_components import LayerNorm, FeedForward
from gpt_model import GPT_CONFIG_MINI


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize KV Cache construction during Prefill & Decode phases "
                    "of GPTModel inference and benchmark speedup vs. uncached decoding."
    )
    parser.add_argument("--prompt", type=str,
                        default="The wonders of self attention in modern language models",
                        help="Seed prompt for Prefill phase.")
    parser.add_argument("--max-new-tokens", type=int, default=12,
                        help="Number of tokens to generate in Decode phase (default: 12).")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Number of Transformer blocks (default: 4).")
    parser.add_argument("--emb-dim", type=int, default=128,
                        help="Embedding dimension (default: 128).")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads (default: 4).")
    return parser.parse_args()


class MultiHeadAttentionWithCache(nn.Module):
    """
    Multi-Head Causal Self-Attention with optional KV Caching.

    - Without cache (`use_cache=False`): recomputes K and V for the full sequence.
    - With cache (`use_cache=True`):
        * Prefill: computes K, V for all prompt tokens and stores them in
          `cache_k` and `cache_v` of shape `(b, num_heads, T_prompt, head_dim)`.
        * Decode: computes q, k, v ONLY for the 1 newest token, appends (k, v)
          to `cache_k` / `cache_v` along dim=2, and attends q against the full cache.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length

        self.WQuery = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WKey = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WValue = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        b, num_tokens, _ = x.shape

        keys_new = self.WKey(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.WQuery(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values_new = self.WValue(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = keys_new
                self.cache_v = values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
                if self.cache_k.shape[2] > self.context_length:
                    self.cache_k = self.cache_k[:, :, -self.context_length:, :]
                    self.cache_v = self.cache_v[:, :, -self.context_length:, :]

            keys = self.cache_k
            values = self.cache_v
            total_len = keys.shape[2]
            pos_start = total_len - num_tokens
            mask_bool = self.mask.bool()[pos_start:total_len, :total_len]
        else:
            keys = keys_new
            values = values_new
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlockWithCache(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionWithCache(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModelWithCache(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([TransformerBlockWithCache(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.current_pos = 0

    def reset_kv_cache(self):
        self.current_pos = 0
        for blk in self.trf_blocks:
            blk.att.reset_cache()

    def forward(self, in_idx, use_cache=False):
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device)

        pos_embeds = self.pos_emb(pos_ids)
        x = self.drop_emb(tok_embeds + pos_embeds)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)
        return self.out_head(x)


def generate_uncached(model, idx, max_new_tokens, context_size):
    """Standard generation without KV cache: recomputes ALL tokens on every step."""
    tokens_processed_per_step = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        tokens_processed_per_step.append(idx_cond.shape[1])
        with torch.no_grad():
            logits = model(idx_cond, use_cache=False)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx, tokens_processed_per_step


def generate_with_kv_cache(model, idx, max_new_tokens, tokenizer=None, verbose=True):
    """
    Two-phase KV-cached generation:
      1. PREFILL: process the entire prompt once, populating K/V caches in all layers.
      2. DECODE : feed ONLY 1 new token per step, appending to K/V caches.
    """
    model.reset_kv_cache()
    n_layers = len(model.trf_blocks)
    tokens_processed_per_step = []
    cache_lengths = []

    # --- PHASE 1: PREFILL ---
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        logits = model(idx, use_cache=True)
    prefill_ms = (time.perf_counter() - t_prefill_start) * 1000.0

    idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)

    sample_k = model.trf_blocks[0].att.cache_k
    sample_v = model.trf_blocks[0].att.cache_v
    kv_bytes = n_layers * (sample_k.numel() + sample_v.numel()) * sample_k.element_size()
    tokens_processed_per_step.append(sample_k.shape[2])
    cache_lengths.append(sample_k.shape[2])

    if verbose and tokenizer is not None:
        prompt_toks = [repr(tokenizer.decode([int(t)])) for t in idx[0, :-1].tolist()]
        first_tok = repr(tokenizer.decode([int(idx_next.item())]))
        print(f"\n{BOLD}{CYAN}{'=' * 78}{RESET}")
        print(f"{BOLD}{CYAN}  PHASE 1: PREFILL (Process Full Prompt in 1 Parallel Pass){RESET}")
        print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")
        print(f"  Prompt tokens ({len(prompt_toks)}): {' '.join(prompt_toks)}")
        print(f"  Input shape      : (1, {len(prompt_toks)})  -->  Q, K, V computed for ALL {len(prompt_toks)} tokens")
        print(f"  K/V cache/layer  : {tuple(sample_k.shape)}  (batch, heads, seq_len, head_dim)")
        print(f"  Total KV memory  : {kv_bytes / 1024:.2f} KB across {n_layers} layers  |  {prefill_ms:.2f} ms")
        print(f"  Predicted token 1: {GREEN}{first_tok}{RESET}\n")

        print(f"{BOLD}{MAGENTA}{'=' * 78}{RESET}")
        print(f"{BOLD}{MAGENTA}  PHASE 2: DECODE (Feed 1 Token/Step, Append to K & V Caches){RESET}")
        print(f"{BOLD}{MAGENTA}{'=' * 78}{RESET}")

    # --- PHASE 2: DECODE ---
    decode_times_ms = []
    for step in range(1, max_new_tokens):
        t_step = time.perf_counter()
        with torch.no_grad():
            logits = model(idx_next, use_cache=True)  # ONLY 1 token fed in!
        step_ms = (time.perf_counter() - t_step) * 1000.0
        decode_times_ms.append(step_ms)

        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

        cur_len = model.trf_blocks[0].att.cache_k.shape[2]
        tokens_processed_per_step.append(1)
        cache_lengths.append(cur_len)

        if verbose and tokenizer is not None:
            in_tok = repr(tokenizer.decode([int(idx[0, -2].item())]))
            out_tok = repr(tokenizer.decode([int(idx_next.item())]))
            slots = f"{CYAN}{'█' * (cur_len - 1)}{GREEN}[+1]{RESET}"
            print(
                f"  Step {step:02d} | Feed {in_tok:<12} (1,1) │ "
                f"q(1×d_k) @ K_cache({cur_len}×d_k)^T │ "
                f"Cache: {slots} ({cur_len}) │ "
                f"Next: {GREEN}{out_tok:<12}{RESET} ({step_ms:.2f} ms)"
            )

    return idx, tokens_processed_per_step, cache_lengths


def plot_kv_cache_growth(uncached_tokens, cached_tokens, cache_lengths,
                         uncached_ms, cached_ms, figsize=(13, 4.8),
                         save_path=None, show=True):
    steps = list(range(len(uncached_tokens)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(steps, uncached_tokens, marker="o", linewidth=2, color="#e03131",
             label="Without KV Cache (tokens recomputed/step)")
    ax1.plot(steps, cached_tokens, marker="s", linewidth=2, color="#2f9e44",
             label="With KV Cache (tokens computed/step)")
    ax1.plot(steps, cache_lengths, linestyle="--", linewidth=1.8, color="#1971c2",
             label="KV Cache Length (stored K/V slots)")
    ax1.set_title("Prefill (Step 0) vs. Decode (Steps 1+): Work vs. Cache Growth")
    ax1.set_xlabel("Generation Step (0 = Prefill, 1+ = Decode)")
    ax1.set_ylabel("Tokens / Cache Slots")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    speedup = uncached_ms / max(cached_ms, 1e-6)
    bars = ax2.bar(["Without KV Cache", "With KV Cache\n(Prefill + Decode)"],
                   [uncached_ms, cached_ms], color=["#e03131", "#2f9e44"])
    ax2.set_ylabel("Total Inference Time (ms)")
    ax2.set_title(f"End-to-End Inference Latency ({speedup:.2f}× Speedup)")
    for bar, val in zip(bars, [uncached_ms, cached_ms]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f} ms",
                 ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    tokenizer = tiktoken.get_encoding("gpt2")

    cfg = dict(GPT_CONFIG_MINI)
    cfg["n_layers"] = args.n_layers
    cfg["emb_dim"] = args.emb_dim
    cfg["n_heads"] = args.n_heads
    cfg["drop_rate"] = 0.0  # deterministic for exact equivalence check

    torch.manual_seed(123)
    model = GPTModelWithCache(cfg)
    model.eval()

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt)).unsqueeze(0)

    # 1. Visual Prefill + Decode KV Cache Walkthrough
    cached_ids, cached_toks, cache_lens = generate_with_kv_cache(
        model, prompt_ids, max_new_tokens=args.max_new_tokens, tokenizer=tokenizer, verbose=True
    )

    # 2. Timing & Exact Equivalence Check vs. Uncached Generation
    t0 = time.perf_counter()
    uncached_ids, uncached_toks = generate_uncached(
        model, prompt_ids, max_new_tokens=args.max_new_tokens, context_size=cfg["context_length"]
    )
    uncached_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    cached_ids_bench, _, _ = generate_with_kv_cache(
        model, prompt_ids, max_new_tokens=args.max_new_tokens, verbose=False
    )
    cached_ms = (time.perf_counter() - t1) * 1000.0

    exact_match = torch.equal(uncached_ids, cached_ids_bench)
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  EQUIVALENCE & SPEEDUP SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"  Exact token match (Uncached == KV Cached): {GREEN}{exact_match}{RESET}")
    print(f"  Total tokens computed — Uncached: {sum(uncached_toks)}  |  KV Cached: {sum(cached_toks)}")
    print(f"  Wall-clock time       — Uncached: {uncached_ms:.2f} ms  |  KV Cached: {cached_ms:.2f} ms "
          f"({uncached_ms / max(cached_ms, 1e-6):.2f}× speedup)")
    assert exact_match, "Cached and uncached outputs must match identically!"

    plot_kv_cache_growth(uncached_toks, cached_toks, cache_lens, uncached_ms, cached_ms)
