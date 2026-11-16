# From Tokenization to Q, K, V Self-Attention (`tokenize_to_Q_K_V`)

A hands-on, step-by-step implementation of the foundational stages of a Large Language Model (LLM) in PyTorch—starting from raw text tokenization and sliding-window data loaders, moving through token and positional embeddings, and culminating in trainable Query-Key-Value ($Q, K, V$) self-attention with interactive training visualizers.

*Inspired by Sebastian Raschka's **Build a Large Language Model (From Scratch)**.*

---

## Setup & Installation

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the Pretrained Word2Vec Cache (One-Time Setup for Files 5-C – 7-D)
Files 6 and 7 use pretrained 300-dimensional Google News Word2Vec vectors (`word2vec-google-news-300`) reduced via SVD/PCA so that attention scores operate on meaningful semantic embeddings even before training.

Run the downloader once to cache `word2vec-google-news-300.kv` and `word2vec-google-news-300.kv.vectors.npy` locally (ignored by Git):
```bash
python word2vec_download.py
```

---

## Learning Progression & File Guide

### Stage 1: Tokenization
| Step | File | Description | How to Run |
| :--- | :--- | :--- | :--- |
| **File 1** | `tokenizer.py` | Builds a regex-based word & punctuation tokenizer (`TokenizerV1`) from scratch on Project Gutenberg's *The Book of Wonders* (`book_of_wonders.txt`), including `<\|unk\|>` and `<\|endoftext\|>` special tokens. | `python tokenizer.py` |
| **File 2** | `bpe_tokenizer.py` | Introduces subword Byte-Pair Encoding (BPE) using OpenAI's `tiktoken` (`gpt2` vocabulary) to handle arbitrary out-of-vocabulary words without `<\|unk\|>`. | `python bpe_tokenizer.py` |

### Stage 2: Training Data Preparation & Embeddings
| Step | File | Description | How to Run |
| :--- | :--- | :--- | :--- |
| **File 3** | `input_for_training.py` | Implements a sliding-window PyTorch `Dataset` (`GPTDatasetV1`) and `DataLoader` (`createDataLoader`) with configurable `max_length` and `stride` to produce `(input, target)` next-token prediction pairs. | `python input_for_training.py` |
| **File 5-A** | `token_embeddings.py` | Demonstrates how token IDs map to dense vectors via `nn.Embedding` (and why one-hot encoding + `nn.Linear` is equivalent), then adds absolute positional embeddings (`embedding_layer.png`). | `python token_embeddings.py` |
| **File 5-B** | `token_embeddings_with_pdf.py` | Extends `token_embeddings.py` to ingest custom PDF documents (via `pypdf` or `langchain`) in addition to `.txt` files, cleaning the extracted text and feeding it through the same `DataLoader` and embedding pipeline. | `python token_embeddings_with_pdf.py --pdf mybook.pdf --pages 1-10`<br>or<br>`python token_embeddings_with_pdf.py --text book_of_wonders.txt` |
| **File 5-C** | `google-news-word2vec.py` | Explores vector arithmetic, cosine similarity, and nearest neighbors in the 300D `word2vec-google-news-300` embedding space. | `python google-news-word2vec.py` |
| **File 5-D** | `word2vec_download.py` | One-time helper script that downloads and saves `word2vec-google-news-300.kv` to disk for fast offline loading in Files 6 and 7. | `python word2vec_download.py` |

### Stage 3: Self-Attention Without Trainable Weights
| Step | File | Description | How to Run |
| :--- | :--- | :--- | :--- |
| **File 6** | `attention_without_trainable_weights.py` | Computes raw dot-product self-attention directly on 3D Word2Vec embeddings ($Q = K = V = X$). Includes interactive 3D vector plots (toggleable input vs. context vectors), an attention weight heatmap, and per-token Shannon entropy analysis. | `python attention_without_trainable_weights.py "the dog chased the cat because it was very hungry"` |

### Stage 4: Self-Attention With Trainable $Q, K, V$ Weights (`File 7-A` – `7-D`)
Introduces learned linear projections ($W_Q, W_K, W_V$) and scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Step | File | What It Adds | How to Run |
| :--- | :--- | :--- | :--- |
| **File 7-A** | `attention_with_Q_K_V.py` | Core `SelfAttentionv1(nn.Module)` implementation with `nn.Linear` projections for $W_Q, W_K, W_V$ and $\sqrt{d_k}$ scaling on 3D Word2Vec embeddings ($d_{\text{in}}=3 \to d_{\text{out}}=2$). | `python attention_with_Q_K_V.py "the dog chased the cat because it was very hungry"` |
| **File 7-B** | `attention_with_Q_K_V_with_visuals.py` | Adds a 2×2 Matplotlib dashboard comparing input embeddings, the learned attention weight matrix, per-word attention entropy (with uniform baseline), and the resulting context vectors. | `python attention_with_Q_K_V_with_visuals.py "the dog chased the cat because it was very hungry"` |
| **File 7-C** | `attention_with_Q_K_V_withPCA.py` | Adds `--pca-init` (`init_qkv_from_pca`) on 3D embeddings ($3\text{D} \to 2\text{D}$) to compare random untrained $W_Q, W_K, W_V$ projections against variance-preserving PCA initialization. | `python attention_with_Q_K_V_withPCA.py "the dog chased the cat because it was very hungry" --pca-init` |
| **File 7-D** | `attention_with_Q_K_V_higherDimensions.py` | Expands input embeddings from 3D to 6D ($d_{\text{in}}=6 \to d_{\text{out}}=5$) so PCA initialization and $Q, K, V$ projections retain richer semantic structure, printing `WQuery.weight` before and after `--pca-init`. | `python attention_with_Q_K_V_higherDimensions.py "the dog chased the cat because it was very hungry" --pca-init` |

### Stage 5: Interactive $Q, K, V$ Training Visualizers (`File 8-A` & `8-B`)
Watch $W_Q, W_K, W_V$ learn step-by-step during backpropagation on a causal next-word prediction task while all other parameters (embeddings, positional encodings, $W_O$, unembedding) are held frozen.

| Step | File | Description | How to Run |
| :--- | :--- | :--- | :--- |
| **File 8-A** | `qkv_train_viz.py` | **Terminal UI visualizer**: Step through training (`[Enter]` for 1 step, `[N]` for $N$ steps, `[d]` to toggle $\Delta$ mode) and watch $W_Q, W_K, W_V$, $Q, K, V$, causal attention weights, and next-word probabilities update live in the terminal. | `python qkv_train_viz.py`<br>or<br>`python qkv_train_viz.py --sentence "the dog chased the ball across the quiet park"` |
| **File 8-B** | `qkv_plot_viz.py` | **Matplotlib interactive dashboard**: Choose **Single-Word** or **Full-Sentence** loss scope at startup, then step through training with `[Space]`/`[Enter]`. Displays live heatmaps of $W_Q, W_K, W_V$, a plain-English breakdown of what the last gradient step changed, and (for short sentences) the exact chain-rule gradients $\frac{\partial L}{\partial W_Q}, \frac{\partial L}{\partial W_K}, \frac{\partial L}{\partial W_V}$ and Adam update arithmetic. | `python qkv_plot_viz.py`<br>or<br>`python qkv_plot_viz.py --sentence "dogs chase cats"` |

### Stage 6: Causal (Masked) Self-Attention (`File 9-A`)
| Step | File | Description | How to Run |
| :--- | :--- | :--- | :--- |
| **File 9-A** | `causal_attention.py` | Introduces `CausalSelfAttention(nn.Module)` so each token can only attend to itself and earlier tokens. Walks through both the naive `torch.tril` + row renormalization method and the standard Transformer `-inf` `masked_fill` before `softmax` (using `register_buffer`), with a 4-panel visual pipeline and side-by-side Unmasked vs. Causal heatmap comparison. | `python causal_attention.py "the dog chased the cat because it was very hungry" --pca-init` |
