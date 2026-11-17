# Ramakrishnan A
# File 13 in the series on Building LLMs from scratch
# End-to-End Pretraining & Inference: training GPTModel (File 12-B) on
# book_of_wonders.txt with AdamW + Cross-Entropy Loss, tracking train/val
# loss & perplexity, and generating text with temperature & top-k sampling.
# Lots of pranams to Sebastian Raschka

import argparse
import math
import os
import sys
import time
import tiktoken
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from input_for_training import createDataLoader
from gpt_model import GPTModel, generate_text_simple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end GPTModel pretraining on text + temperature/top-k "
                    "next-token inference."
    )
    parser.add_argument("--text", type=str, default=os.path.join(SCRIPT_DIR, "book_of_wonders.txt"),
                        help="Path to the training .txt file (default: book_of_wonders.txt).")
    parser.add_argument("--max-chars", type=int, default=40000,
                        help="Number of characters to use for fast CPU training demo "
                             "(0 = use entire file; default: 40000).")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default: 4).")
    parser.add_argument("--context-length", type=int, default=64,
                        help="Context window length in tokens (default: 64).")
    parser.add_argument("--lr", type=float, default=4e-4,
                        help="AdamW learning rate (default: 4e-4).")
    parser.add_argument("--eval-freq", type=int, default=10,
                        help="Evaluate train/val loss every N optimizer steps (default: 10).")
    parser.add_argument("--prompt", type=str, default="The wonders of",
                        help="Prompt for before/after training inference comparison.")
    parser.add_argument("--max-new-tokens", type=int, default=20,
                        help="Number of tokens to generate during inference (default: 20).")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (>0 for stochastic, 0.0 for greedy; default: 0.8).")
    parser.add_argument("--top-k", type=int, default=25,
                        help="Top-k sampling cutoff (0 disables top-k; default: 25).")
    parser.add_argument("--save-checkpoint", type=str, default="",
                        help="Optional path to save trained model weights (.pt).")
    parser.add_argument("--no-tui", action="store_true",
                        help="Disable the live ANSI terminal graphic dashboard and print plain logs.")
    return parser.parse_args()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)  # shape: (1, seq_len)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # Flatten (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    Autoregressive decoding with optional top-k filtering and temperature
    scaling (Ch 5.3):
      - temperature == 0.0 -> deterministic greedy argmax decoding
      - temperature > 0.0  -> scales logits by 1/temperature and samples via
                              torch.multinomial (optionally restricted to top_k)
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_sample(model, tokenizer, device, start_context, context_size, max_new_tokens=20):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer).replace("\n", " ")
    print(f"  Sample -> {repr(decoded_text)}")
    model.train()


# ---------------------------------------------------------------------------
# Live Command-Line Training Graphic (pure ANSI 256-color, zero extra deps)
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[94m"


def _sparkline(values, width=32):
    if not values:
        return "(waiting for first eval)"
    blocks = "▁▂▃▄▅▆▇█"
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    span = hi - lo if hi > lo else 1e-6
    return "".join(
        blocks[min(len(blocks) - 1, int((v - lo) / span * (len(blocks) - 1)))]
        for v in tail
    )


def _progress_bar(cur, total, width=28):
    ratio = min(1.0, max(0.0, cur / max(1, total)))
    filled = int(round(ratio * width))
    return f"{GREEN}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET} {ratio * 100:5.1f}%"


def _compute_grad_telemetry(model):
    """Compute global grad L2 norm and per-component mean |grad| for live display."""
    buckets = {"tok_emb": [], "attn (QKV/O)": [], "mlp (FFN)": [], "out_head": []}
    total_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.detach()
        total_sq += float(g.pow(2).sum().item())
        mean_abs = float(g.abs().mean().item())
        if "tok_emb" in name:
            buckets["tok_emb"].append(mean_abs)
        elif ".att." in name:
            buckets["attn (QKV/O)"].append(mean_abs)
        elif ".ff." in name:
            buckets["mlp (FFN)"].append(mean_abs)
        elif "out_head" in name:
            buckets["out_head"].append(mean_abs)
    summary = {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}
    return math.sqrt(total_sq), summary


def _get_top_next_tokens(model, tokenizer, device, prompt, context_size, k=5):
    was_training = model.training
    model.eval()
    ids = text_to_token_ids(prompt, tokenizer).to(device)[:, -context_size:]
    with torch.no_grad():
        logits = model(ids)[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_p, top_id = torch.topk(probs, k)
    if was_training:
        model.train()
    return [
        (repr(tokenizer.decode([tid.item()])), float(p.item()))
        for tid, p in zip(top_id, top_p)
    ]


def render_training_dashboard(
    epoch, num_epochs, batch_idx, num_batches, global_step, total_steps,
    tokens_seen, elapsed_s, batch_loss, train_loss, val_loss,
    batch_losses, grad_norm, grad_buckets, top_tokens, sample_text,
    batch_shape, vocab_size, prompt, first_frame=False,
):
    """Draw a live, in-place terminal graphic showing what is happening inside GPT."""
    tok_per_sec = tokens_seen / max(elapsed_s, 1e-6)
    ppl = math.exp(min(batch_loss, 20.0))
    val_str = f"{val_loss:.3f} (ppl {math.exp(min(val_loss, 20.0)):.1f})" if val_loss is not None else "pending..."
    tr_str = f"{train_loss:.3f}" if train_loss is not None else f"{batch_loss:.3f}"

    max_bucket = max(max(grad_buckets.values(), default=1e-9), 1e-9)

    lines = [
        f"{BOLD}{CYAN}┌──────────────────────────────────────────────────────────────────────────────┐{RESET}",
        f"{BOLD}{CYAN}│  GPTModel Live Pretraining Monitor  (Batch {batch_shape} -> Vocab {vocab_size:,})  │{RESET}",
        f"{BOLD}{CYAN}└──────────────────────────────────────────────────────────────────────────────┘{RESET}",
        f"  {BOLD}Progress{RESET} : Epoch {YELLOW}{epoch}/{num_epochs}{RESET}  Batch {batch_idx}/{num_batches}  "
        f"Step {global_step + 1}/{total_steps}  {_progress_bar(global_step + 1, total_steps)}",
        f"  {BOLD}Throughput{RESET}: {tokens_seen:,} tokens seen  |  {tok_per_sec:,.0f} tok/s  |  {elapsed_s:.1f}s elapsed",
        "",
        f"  {BOLD}Pipeline{RESET} : {GREEN}[DataLoader]{RESET} ─▶ {GREEN}[GPTModel Forward]{RESET} ─▶ "
        f"{YELLOW}[CrossEntropy Loss]{RESET} ─▶ {MAGENTA}[Autograd Backward]{RESET} ─▶ {CYAN}[AdamW Step]{RESET}",
        f"  {BOLD}Loss{RESET}     : Batch {YELLOW}{batch_loss:.3f}{RESET} (ppl {ppl:,.1f})  |  "
        f"Eval Train {GREEN}{tr_str}{RESET}  |  Eval Val {CYAN}{val_str}{RESET}",
        f"  {BOLD}Trend{RESET}    : {BLUE}{_sparkline(batch_losses, width=48)}{RESET}  (latest {batch_loss:.3f})",
        "",
        f"  {BOLD}Gradient Flow{RESET} (Global L2 Norm = {MAGENTA}{grad_norm:.3f}{RESET}):",
    ]

    for comp, val in grad_buckets.items():
        bar_len = int(round((val / max_bucket) * 24))
        lines.append(
            f"    {comp:<14} {MAGENTA}{'▓' * bar_len}{DIM}{'░' * (24 - bar_len)}{RESET}  {val:.6f}"
        )

    lines.append("")
    lines.append(f"  {BOLD}Live Next-Token Predictions for {repr(prompt)}{RESET}:")
    for tok_str, prob in top_tokens:
        pbar = int(round(min(1.0, prob * 5.0) * 20))
        lines.append(
            f"    {tok_str:<16} {CYAN}{'█' * pbar}{DIM}{'░' * (20 - pbar)}{RESET}  {prob * 100:5.2f}%"
        )

    lines.append(f"  {BOLD}Latest Sample{RESET}: {DIM}{sample_text[:72]}{RESET}")

    output = "\n".join(lines)
    if first_frame:
        sys.stdout.write("\033[2J\033[H" + output + "\n")
    else:
        sys.stdout.write("\033[H\033[J" + output + "\n")
    sys.stdout.flush()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context,
                       tokenizer, context_size, use_tui=True):
    train_losses, val_losses, track_tokens_seen = [], [], []
    batch_losses = []
    tokens_seen, global_step = 0, -1
    num_batches = len(train_loader)
    total_steps = num_epochs * num_batches
    t0 = time.time()

    latest_train_loss = None
    latest_val_loss = None
    latest_sample = "(generating at end of epoch 1...)"
    latest_top_tokens = _get_top_next_tokens(
        model, tokenizer, device, start_context, context_size
    ) if use_tui else []

    for epoch in range(num_epochs):
        model.train()
        for b_idx, (input_batch, target_batch) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            grad_norm, grad_buckets = _compute_grad_telemetry(model)
            optimizer.step()

            batch_loss_val = float(loss.item())
            batch_losses.append(batch_loss_val)
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                latest_train_loss, latest_val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(latest_train_loss)
                val_losses.append(latest_val_loss)
                track_tokens_seen.append(tokens_seen)
                if use_tui:
                    latest_top_tokens = _get_top_next_tokens(
                        model, tokenizer, device, start_context, context_size
                    )
                else:
                    print(
                        f"  Ep {epoch + 1:02d} (Step {global_step:04d}) | "
                        f"Train loss {latest_train_loss:.3f} (ppl {math.exp(min(latest_train_loss, 20)):.1f}) | "
                        f"Val loss {latest_val_loss:.3f} (ppl {math.exp(min(latest_val_loss, 20)):.1f})"
                    )

            if use_tui:
                render_training_dashboard(
                    epoch=epoch + 1,
                    num_epochs=num_epochs,
                    batch_idx=b_idx,
                    num_batches=num_batches,
                    global_step=global_step,
                    total_steps=total_steps,
                    tokens_seen=tokens_seen,
                    elapsed_s=time.time() - t0,
                    batch_loss=batch_loss_val,
                    train_loss=latest_train_loss,
                    val_loss=latest_val_loss,
                    batch_losses=batch_losses,
                    grad_norm=grad_norm,
                    grad_buckets=grad_buckets,
                    top_tokens=latest_top_tokens,
                    sample_text=latest_sample,
                    batch_shape=tuple(input_batch.shape),
                    vocab_size=tokenizer.n_vocab,
                    prompt=start_context,
                    first_frame=(global_step == 0),
                )

        # Update end-of-epoch sample
        model.eval()
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model,
                idx=encoded,
                max_new_tokens=20,
                context_size=context_size,
            )
        latest_sample = token_ids_to_text(token_ids, tokenizer).replace("\n", " ")
        model.train()

        if not use_tui:
            print(f"End of Epoch {epoch + 1}:")
            print(f"  Sample -> {repr(latest_sample)}")

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
                figsize=(10, 4.8), save_path=None, show=True):
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(epochs_seen, train_losses, label="Training loss", linewidth=2.0, color="#1f77b4")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", linewidth=2.0, color="#ff7f0e")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Seen")

    plt.title("GPTModel Pretraining: Training vs. Validation Loss")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    with open(args.text, "r", encoding="utf-8") as f:
        raw_text = f.read()
    if args.max_chars > 0:
        raw_text = raw_text[:args.max_chars]

    print("=" * 78)
    print("STEP 1: Prepare Train / Validation DataLoaders (90% / 10% Split)")
    print("=" * 78)
    split_idx = int(0.90 * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]

    train_loader = createDataLoader(
        train_data,
        batch_size=args.batch_size,
        max_length=args.context_length,
        stride=args.context_length,
        shuffle=True,
        drop_last=True,
    )
    val_loader = createDataLoader(
        val_data,
        batch_size=args.batch_size,
        max_length=args.context_length,
        stride=args.context_length,
        shuffle=False,
        drop_last=False,
    )
    print(f"Text source    : {os.path.basename(args.text)} ({len(raw_text):,} chars)")
    print(f"Train batches  : {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"Device         : {device}\n")

    print("=" * 78)
    print("STEP 2: Instantiate GPTModel & Inspect Untrained Baseline")
    print("=" * 78)
    cfg = {
        "vocab_size": tokenizer.n_vocab,
        "context_length": args.context_length,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    torch.manual_seed(123)
    model = GPTModel(cfg).to(device)
    init_train_loss, init_val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter=5)
    print(f"Parameters          : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial Train Loss  : {init_train_loss:.3f} (expected ~ ln(50257) = {math.log(50257):.2f})")
    print(f"Initial Val Loss    : {init_val_loss:.3f}")
    print("Before training:")
    generate_and_print_sample(model, tokenizer, device, args.prompt, args.context_length, args.max_new_tokens)

    print("\n" + "=" * 78)
    print(f"STEP 3: Train with AdamW (epochs={args.epochs}, lr={args.lr})")
    print("=" * 78)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    use_tui = (not args.no_tui) and sys.stdout.isatty()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=5,
        start_context=args.prompt,
        tokenizer=tokenizer,
        context_size=args.context_length,
        use_tui=use_tui,
    )

    epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    print("\n" + "=" * 78)
    print("STEP 4: Inference Comparison (Greedy vs. Temperature + Top-k Sampling)")
    print("=" * 78)
    model.eval()
    prompt_ids = text_to_token_ids(args.prompt, tokenizer).to(device)

    torch.manual_seed(42)
    greedy_ids = generate(
        model, prompt_ids, max_new_tokens=args.max_new_tokens,
        context_size=args.context_length, temperature=0.0,
    )
    print(f"Greedy (temp=0.0)             : {repr(token_ids_to_text(greedy_ids, tokenizer))}")

    torch.manual_seed(42)
    sampled_ids = generate(
        model, prompt_ids, max_new_tokens=args.max_new_tokens,
        context_size=args.context_length, temperature=args.temperature, top_k=args.top_k,
    )
    print(f"Sampled (temp={args.temperature}, top_k={args.top_k}): {repr(token_ids_to_text(sampled_ids, tokenizer))}")

    if args.save_checkpoint:
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"\nSaved model checkpoint to: {args.save_checkpoint}")
