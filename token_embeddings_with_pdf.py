# Ramakrishnan A
# File 5-B in the series on Building LLMs from scratch
# Lots of pranams to Sebastian Raschka

# Embedding layer Weight Matrix
# Token IDs are rows
# Dimensions/Features are columns

# Embedding Layer Weight Matrix is optimized using Backpropogation

# For now, we will only create the embedding layer matrix

# ---------------------------------------------------------------
# Update: PDF importer
# The training text can now come from a PDF instead of book_of_wonders.txt.
# The PDF text is extracted (pypdf or LangChain), cleaned up, and then goes
# through exactly the same DataLoader -> token embedding -> positional
# embedding steps as before.
#
# How to run:
#   python token_embeddings_with_pdf.py                      -> asks PDF or txt, then opens a file picker
#   python token_embeddings_with_pdf.py --pdf                -> file picker straight away
#   python token_embeddings_with_pdf.py --pdf mybook.pdf     -> use this PDF
#   python token_embeddings_with_pdf.py --pdf mybook.pdf --pages 1-10
#   python token_embeddings_with_pdf.py --pdf mybook.pdf --loader langchain
#   python token_embeddings_with_pdf.py --text book_of_wonders.txt   -> the old behaviour
#
# Install:  pip install pypdf
#           pip install langchain-community pypdf    (only if you want --loader langchain)
# ---------------------------------------------------------------

from operator import pos

import argparse
import glob
import os
import re

import torch
import tiktoken
from transformers import GPT2Model
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from input_for_training import createDataLoader
#matplotlib.use("Agg")

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


# ===============================================================
# PDF importer
# ===============================================================

def choose_pdf_file(start_dir="."):
    """
    Let the user pick a PDF.
    1. Try a normal file-open dialog (tkinter).
    2. If there is no dialog (no display / tkinter missing) or it is cancelled,
       list the PDFs in start_dir and ask in the terminal.
    Returns the chosen path, or None if nothing was chosen.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()                     # hide the empty main window
        root.attributes("-topmost", True)   # bring the dialog to the front
        root.update()
        path = filedialog.askopenfilename(
            title="Choose a PDF to build embeddings from",
            initialdir=os.path.abspath(start_dir),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            return path
        print("No file chosen in the dialog.")
    except Exception as e:
        print(f"(File dialog not available: {e})")

    # ---- terminal fallback ----
    pdfs = sorted(glob.glob(os.path.join(start_dir, "*.pdf")))
    if pdfs:
        print(f"\nPDFs in {os.path.abspath(start_dir)}:")
        for i, p in enumerate(pdfs, 1):
            print(f"  [{i}] {os.path.basename(p)}")
        prompt = "Type a number from the list, or a path to a PDF (Enter to cancel): "
    else:
        prompt = "Type the path to a PDF (Enter to cancel): "

    choice = input(prompt).strip().strip('"').strip("'")
    if not choice:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(pdfs):
        return pdfs[int(choice) - 1]
    return os.path.expanduser(choice)


def parse_page_range(spec, num_pages):
    """
    Turn a 1-based page spec into 0-based page indices.
      None      -> every page
      "3"       -> [2]
      "2-5"     -> [1, 2, 3, 4]
      "1,4,7-9" -> [0, 3, 6, 7, 8]
      "10-"     -> page 10 to the end
    """
    if not spec:
        return list(range(num_pages))

    pages = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start) if start.strip() else 1
            end = int(end) if end.strip() else num_pages
            pages.extend(range(start - 1, end))
        else:
            pages.append(int(part) - 1)

    # keep only real pages, drop duplicates, keep the order given
    seen, result = set(), []
    for p in pages:
        if 0 <= p < num_pages and p not in seen:
            seen.add(p)
            result.append(p)
    if not result:
        raise ValueError(f"--pages '{spec}' selects no pages (the PDF has {num_pages} pages)")
    return result


def load_pdf_pages_pypdf(pdf_path, page_spec=None):
    """Extract text page by page with pypdf. Returns (list_of_page_texts, total_pages)."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError(f"{pdf_path} is password-protected")

    total = len(reader.pages)
    indices = parse_page_range(page_spec, total)
    return [reader.pages[i].extract_text() or "" for i in indices], total


def load_pdf_pages_langchain(pdf_path, page_spec=None):
    """Extract text page by page with LangChain's PyPDFLoader (one Document per page)."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        from langchain.document_loaders import PyPDFLoader   # older LangChain versions

    docs = PyPDFLoader(pdf_path).load()
    total = len(docs)
    indices = parse_page_range(page_spec, total)
    return [docs[i].page_content for i in indices], total


def clean_pdf_text(text):
    """
    PDF text comes out with a line break at the end of every printed line and
    words split with hyphens. Tidy it into flowing paragraphs so the tokenizer
    sees normal prose instead of lots of "\\n" tokens.
    """
    text = text.replace("­", "")                     # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # re-join "embed-\nding" -> "embedding"
    text = re.sub(r"[ \t]+", " ", text)                    # runs of spaces/tabs -> one space
    text = re.sub(r" *\n *", "\n", text)                   # trim spaces around line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)           # single line break inside a paragraph -> space
    text = re.sub(r"\n{3,}", "\n\n", text)                 # keep at most one blank line
    return text.strip()


def load_pdf_text(pdf_path, loader="pypdf", page_spec=None, clean=True):
    """Read a PDF and return its text as one string, ready for createDataLoader."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"No such file: {pdf_path}")

    if loader == "langchain":
        pages, total = load_pdf_pages_langchain(pdf_path, page_spec)
    else:
        pages, total = load_pdf_pages_pypdf(pdf_path, page_spec)

    if clean:
        pages = [clean_pdf_text(p) for p in pages]
    text = "\n\n".join(p for p in pages if p.strip())

    print(f"\nLoaded PDF: {os.path.basename(pdf_path)}  (loader: {loader})")
    print(f"Pages read: {len(pages)} of {total}")
    print(f"Characters extracted: {len(text):,}")

    if not text.strip():
        raise ValueError(
            "No text could be extracted. This PDF is probably scanned images; "
            "it needs OCR (for example ocrmypdf) before it can be tokenized."
        )
    return text


def get_training_text(args):
    """Decide where the text comes from (PDF or .txt) and return it."""
    source = None
    if args.pdf is not None:
        source = "pdf"
    elif args.text is not None:
        source = "text"
    else:
        answer = input("Build embeddings from (1) a PDF or (2) book_of_wonders.txt? [1]: ").strip()
        source = "text" if answer == "2" else "pdf"

    if source == "text":
        text_path = args.text or "book_of_wonders.txt"
        with open(text_path, "r", encoding="utf-8") as f:
            return f.read(), text_path

    pdf_path = args.pdf or choose_pdf_file()
    if not pdf_path:
        raise SystemExit("No PDF chosen. Exiting.")
    text = load_pdf_text(pdf_path, loader=args.loader, page_spec=args.pages, clean=not args.no_clean)

    if args.save_text:
        with open(args.save_text, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved the extracted text to {args.save_text}")

    return text, pdf_path


def get_args():
    parser = argparse.ArgumentParser(description="Token + positional embeddings from a PDF or a text file")
    parser.add_argument("--pdf", nargs="?", const="", default=None,
                        help="Path to a PDF. Give --pdf with no path to open a file picker.")
    parser.add_argument("--text", default=None,
                        help="Path to a .txt file (the original behaviour)")
    parser.add_argument("--loader", choices=["pypdf", "langchain"], default="pypdf",
                        help="Library used to read the PDF (default: pypdf)")
    parser.add_argument("--pages", default=None,
                        help='Pages to read, 1-based, e.g. "1-10" or "1,3,7-9". Default: all pages')
    parser.add_argument("--no-clean", action="store_true",
                        help="Keep the raw PDF text (line breaks, hyphenation) instead of tidying it")
    parser.add_argument("--save-text", default=None,
                        help="Also save the extracted PDF text to this .txt file, to see what the tokenizer sees")
    return parser.parse_args()


# Load the tokenizer
# Using the token set of GPT2
# We want to use the token IDs from this tokenizer and not the weights
tokenizer = tiktoken.get_encoding("gpt2")
#ids = tokenizer.encode("The luggage that the exhausted tourists forgot to retrieve from the carousel after their flight landed in Rome was lost.")
ids = tokenizer.encode("Cats that scare mice chase birds, but mice scare elephants while cats don't.")
#ids = tokenizer.encode("Cats hate dogs but love Naman")
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
#print(embedding_layer(torch.tensor([10])))

"""
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
token_id = 2   # <- change this to any token id in [0, vocab_size)
x = torch.zeros(vocab_size)
x[token_id] = 1.0

y = visualize_linear_layer(linear_weights, x, save_path="embedding_layer.png")
y_minus_bias = torch.tensor(y) - linear_weights.bias.detach()
print("y - bias                     =", y_minus_bias)
print("embedding_layer.weight[token_id] =", embedding_layer.weight[token_id])
print("Match (within float tolerance)?", torch.allclose(y_minus_bias, embedding_layer.weight[token_id], atol=1e-4))
"""

print("Adding Positional embeddings")
print("Moving to GPT 2 sizes")
vocabulary_size = 50257
embedding_training_dimensions_with_pos = 4

with_pos_embedding_layer = torch.nn.Embedding(vocabulary_size, embedding_training_dimensions_with_pos)
print("\nEmbedding Layer Weights")
#print(with_pos_embedding_layer.weight[:3])
print(with_pos_embedding_layer.weight.shape)
print(with_pos_embedding_layer.num_embeddings)
print(with_pos_embedding_layer.embedding_dim)

if __name__ == "__main__":
    args = get_args()

    # Load the text -- from a PDF (new) or from a .txt file (as before)
    _text, source_name = get_training_text(args)

    max_length = 4
    batch_size = 6

    # Make sure there is enough text for one full batch of (batch_size x max_length) tokens
    num_tokens = len(tokenizer.encode(_text, allowed_special={"<|endoftext|>"}))
    print(f"\nSource: {source_name}")
    print(f"Number of tokens in the text: {num_tokens:,}")
    min_tokens = max_length + (batch_size - 1) * max_length + 1
    if num_tokens < min_tokens:
        print(f"Warning: {num_tokens} tokens may be too few for one batch "
              f"(needs about {min_tokens} for batch_size={batch_size}, max_length={max_length}).")

    dataloader = createDataLoader(_text, batch_size=batch_size, max_length=max_length, stride=max_length, shuffle=False)

    data = iter(dataloader)
    try:
        inputs, targets = next(data)
    except StopIteration:
        raise SystemExit("The DataLoader produced no batches -- the text is too short. "
                         "Try more pages or a smaller batch_size.")

    print("Token IDs: \n", inputs)
    print("\nInputs shape: ", inputs.shape)
    print("First row as text:", repr(tokenizer.decode(inputs[0].tolist())))

    print(f"Starting Conversion of the tokens into {embedding_training_dimensions_with_pos} dimension vectors")
    vector_embeddings = with_pos_embedding_layer(inputs)
    print("Shape of the Vector embeddings" , vector_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, embedding_training_dimensions_with_pos)

    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print('Shape of Positional Embeddings Matrix:', pos_embeddings.shape)

    final_vector_embeddings = vector_embeddings + pos_embeddings

    print("Shape of Final Vector Embeddings", final_vector_embeddings.shape)
    print("Initial Vector Embedding\n", vector_embeddings[1])
    print("Initial Positional Embedding\n", pos_embeddings)
    print("Final Embedding\n", final_vector_embeddings[1])
