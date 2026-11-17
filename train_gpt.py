# Ramakrishnan A
# File 13 in the series on Building LLMs from scratch
# End-to-End Pretraining & Inference: training GPTModel (File 12-B) on
# book_of_wonders.txt with AdamW + Cross-Entropy Loss, tracking train/val
# loss & perplexity, and generating text with temperature & top-k sampling.
# Lots of pranams to Sebastian Raschka

import argparse
import math
import os
import re
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
    parser.add_argument("--pdf", type=str, default="",
                        help="Optional path to a PDF file to train on. If omitted or not found, "
                             "falls back to --text (book_of_wonders.txt).")
    parser.add_argument("--pages", type=str, default="",
                        help="Optional page range for --pdf (e.g. '1-10').")
    parser.add_argument("--text", type=str, default=os.path.join(SCRIPT_DIR, "book_of_wonders.txt"),
                        help="Path to the fallback .txt file (default: book_of_wonders.txt).")
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
    parser.add_argument("--trace-steps", type=int, default=5,
                        help="Number of next-token prediction steps to visualize in detail (default: 5).")
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


def plot_prediction_walkthrough(trace_steps, prompt, figsize=(13, 2.4), save_path=None, show=True):
    """
    Matplotlib graphic showing step-by-step how autoregressive next-token
    prediction works: each row is one generated token, showing the context
    so far and the top-6 candidate probabilities with the chosen token in green.
    """
    if not trace_steps:
        return
    n_steps = len(trace_steps)
    fig, axes = plt.subplots(n_steps, 1, figsize=(figsize[0], figsize[1] * n_steps))
    if n_steps == 1:
        axes = [axes]

    for step_idx, step in enumerate(trace_steps):
        ax = axes[step_idx]
        cands = step["candidates"]
        labels = [c[0] for c in cands][::-1]
        probs = [c[2] * 100.0 for c in cands][::-1]
        colors = ["#2f9e44" if c[3] else "#4c6ef5" for c in cands][::-1]

        bars = ax.barh(range(len(labels)), probs, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(0, max(max(probs) * 1.25, 1.0))
        ax.set_xlabel("Probability (%)" if step_idx == n_steps - 1 else "")
        ctx_preview = step["context_text"].replace("\n", "\\n")
        if len(ctx_preview) > 55:
            ctx_preview = "..." + ctx_preview[-52:]
        ax.set_title(
            f"Step {step_idx + 1}: Context {repr(ctx_preview)}  →  Picked {step['chosen_str']} ({step['chosen_prob'] * 100:.2f}%)",
            fontsize=9.5, loc="left", fontweight="bold"
        )
        for bar, p in zip(bars, probs):
            ax.text(p, bar.get_y() + bar.get_height() / 2, f" {p:.2f}%",
                    va="center", ha="left", fontsize=8)

    plt.suptitle(f"How Autoregressive Next-Token Prediction Happens (Prompt: {repr(prompt)})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def generate_with_visual_trace(model, tokenizer, device, prompt, context_size,
                               steps=5, temperature=0.8, top_k=25):
    """
    Run `steps` autoregressive prediction steps, printing a step-by-step
    command-line graphic of:
      Context IDs -> GPTModel Logits (50,257) -> Top-k & Temperature -> Softmax -> Picked Token
    Returns the trace list for optional Matplotlib plotting.
    """
    model.eval()
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    trace = []

    print("\n" + BOLD + CYAN + "=" * 78 + RESET)
    print(BOLD + CYAN + "  HOW NEXT-TOKEN PREDICTION ACTUALLY HAPPENS (STEP-BY-STEP TRACE)" + RESET)
    print(BOLD + CYAN + "=" * 78 + RESET)
    mode_desc = "Greedy (argmax)" if temperature == 0.0 else f"Stochastic (temp={temperature}, top_k={top_k})"
    print(f"  Prompt: {repr(prompt)}  |  Decoding: {mode_desc}\n")

    for step_idx in range(steps):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]  # (1, vocab_size)

        raw_logits = logits.clone()
        filtered_logits = logits.clone()
        if top_k is not None and top_k > 0:
            top_vals, _ = torch.topk(filtered_logits, top_k)
            min_val = top_vals[:, -1].unsqueeze(-1)
            filtered_logits = torch.where(
                filtered_logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                filtered_logits,
            )

        if temperature > 0.0:
            scaled_logits = filtered_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            probs = torch.softmax(filtered_logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        chosen_id = int(idx_next.item())
        chosen_str = repr(tokenizer.decode([chosen_id]))
        chosen_prob = float(probs[0, chosen_id].item())

        # Collect top-6 candidates for the graphic (plus the picked token if outside top-6)
        top6_p, top6_ids = torch.topk(probs[0], 6)
        top6_id_set = set(int(t.item()) for t in top6_ids)
        display_ids = list(top6_ids)
        if chosen_id not in top6_id_set:
            display_ids.append(torch.tensor(chosen_id, device=logits.device))

        candidates = []
        for tid in display_ids:
            t_int = int(tid.item())
            candidates.append((
                repr(tokenizer.decode([t_int])),
                float(raw_logits[0, t_int].item()),
                float(probs[0, t_int].item()),
                t_int == chosen_id,
            ))

        ctx_tokens = [repr(tokenizer.decode([int(t)])) for t in idx_cond[0].tolist()[-6:]]
        ctx_text = token_ids_to_text(idx, tokenizer)

        # Print CLI graphic for this step
        print(f"  {BOLD}┌─ Step {step_idx + 1}/{steps} ──────────────────────────────────────────────────────────{RESET}")
        print(f"  {BOLD}│{RESET} Context window (last {len(idx_cond[0])} tokens): {' '.join(ctx_tokens)}")
        print(f"  {BOLD}│{RESET} Forward pass: GPTModel -> logits shape (1, 50257) -> Top-k & Softmax:")
        max_p = max(max(c[2] for c in candidates), 1e-6)
        for tok_s, raw_l, p_val, is_picked in candidates:
            bar_w = int(round((p_val / max_p) * 20))
            marker = f"{GREEN}◀ PICKED{RESET}" if is_picked else ""
            col = GREEN if is_picked else CYAN
            print(f"  {BOLD}│{RESET}   {tok_s:<14} logit {raw_l:+6.2f} │ {col}{'█' * bar_w}{DIM}{'░' * (20 - bar_w)}{RESET} {p_val * 100:5.2f}% {marker}")
        print(f"  {BOLD}└─▶ Appended {GREEN}{chosen_str}{RESET} (new text: {repr(ctx_text + tokenizer.decode([chosen_id]))})\n")

        trace.append({
            "context_text": ctx_text,
            "candidates": candidates,
            "chosen_str": chosen_str,
            "chosen_prob": chosen_prob,
        })
        idx = torch.cat((idx, idx_next), dim=1)

    return trace


def _extract_pdf_text(pdf_path, page_spec=""):
    """Extract and clean text from a PDF using pypdf, with optional page range (e.g. '1-10')."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    if page_spec:
        indices = []
        for part in page_spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                s, e = part.split("-", 1)
                s_i = int(s) if s.strip() else 1
                e_i = int(e) if e.strip() else total
                indices.extend(range(s_i - 1, e_i))
            else:
                indices.append(int(part) - 1)
        indices = [i for i in indices if 0 <= i < total]
    else:
        indices = list(range(total))

    raw = "\n\n".join((reader.pages[i].extract_text() or "") for i in indices)
    raw = raw.replace("\u00ad", "")
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    raw = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip(), len(indices), total


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    raw_text = None
    source_label = os.path.basename(args.text)
    if args.pdf:
        if os.path.isfile(args.pdf):
            raw_text, _pages_used, _total_pages = _extract_pdf_text(args.pdf, args.pages)
            source_label = f"{os.path.basename(args.pdf)} (pages {_pages_used}/{_total_pages})"
        else:
            print(f"[Notice] PDF not found at {args.pdf!r} -- falling back to {args.text}")

    if raw_text is None:
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
    print(f"Text source    : {source_label} ({len(raw_text):,} chars)")
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

    torch.manual_seed(42)
    trace = generate_with_visual_trace(
        model, tokenizer, device,
        prompt=args.prompt,
        context_size=args.context_length,
        steps=args.trace_steps,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    plot_prediction_walkthrough(trace, args.prompt)

    if args.save_checkpoint:
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"\nSaved model checkpoint to: {args.save_checkpoint}")
