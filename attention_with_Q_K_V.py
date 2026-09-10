# Ramakrishnan A
# File 7 in the series on Building LLMs from scratch
# Lots of pranams to Sebastian Raschka

# Self-attention with trainable Query, Key, and Value weight matrices.
# Unlike File 6 (where Q = K = V = inputs directly), here each of Q, K, V
# is produced by its own learned linear projection, so they can genuinely
# diverge from one another once trained.

import torch.nn as nn
import torch
import argparse
import re

# Reuse the word2vec-backed embedding generator (with local caching) from
# File 6, rather than duplicating it here. That file's execution code is
# wrapped in `if __name__ == "__main__":`, so importing it only pulls in
# the function/constant definitions -- no side effects.
from attention_without_trainable_weights import generate_semantic_embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-attention with trainable Q/K/V weights, using word2vec "
                     "embeddings for the input sentence."
    )
    parser.add_argument("sentence", type=str, help="The sentence to tokenize and embed")
    return parser.parse_args()


# Original hardcoded example, kept for reference. Replaced below by a
# sentence typed on the command line, embedded via word2vec instead.
'''
input_words = ["the", "dog", "chased", 'the', 'cat', 'because', 'it', 'was', 'very', 'hungry']

inputs = torch.tensor([
    [0.10, 0.10, 0.10], #The
    [0.85, 0.75, 0.65], #dog
    [0.70, 0.65, 0.75], #chased
    [0.10, 0.10, 0.10], #the
    [0.78, 0.70, 0.62], #cat
    [0.30, 0.25, 0.35], #because
    [0.72, 0.68, 0.60], #it
    [0.25, 0.20, 0.30], #was
    [0.35, 0.40, 0.30], #very
    [0.65, 0.75, 0.55]  #hungry
])

d_in = inputs.shape[1]
d_out = inputs.shape[1] - 1
'''


class SelfAttentionv1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # Each of Q, K, V gets its OWN learned linear projection -- the
        # key upgrade over File 6. A word's query, key, and value can end
        # up genuinely different from each other once these are trained,
        # instead of all three being the same raw embedding.
        '''
        self.WQuery = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.WKey = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.WValue = torch.nn.Parameter(torch.rand(d_in, d_out))
        '''
        self.WQuery = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WKey = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.WValue = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # Project the input into Query, Key, and Value spaces.
        '''
        queries = inputs @ self.WQuery
        keys = inputs @ self.WKey
        values = inputs @ self.WValue
        '''
        queries = self.WQuery(x)
        keys = self.WKey(x)
        values = self.WValue(x)

        # Raw attention scores: how much does each word's query match
        # every other word's key? Shape: (n_words, n_words).
        attentionScores = queries @ keys.T

        # Scale by sqrt(d_k) before softmax. Without this, larger d_k
        # values make dot products grow in magnitude, pushing softmax
        # toward extreme outputs and destabilizing training -- this is
        # the same scaling used in the original "Attention Is All You
        # Need" paper.
        d_k = keys.shape[1]
        attentionWeights = torch.softmax(attentionScores / (d_k**0.5), dim=-1)

        # Each word's final context vector: a weighted blend of every
        # word's VALUE, weighted by how much attention that word received.
        context_vector = attentionWeights @ values

        return context_vector


if __name__ == "__main__":
    args = parse_args()
    input_words = re.findall(r"\b[\w']+\b", args.sentence)

    print("Sentence Used:", args.sentence)
    print("Words:", input_words)

    # Real, word2vec-based semantic embeddings for the sentence, reduced
    # to 3D -- replaces the old hardcoded example above.
    inputs = generate_semantic_embeddings(input_words)
    print("Embeddings (from word2vec, reduced to 3D):\n", inputs)

    d_in = inputs.shape[1]
    d_out = inputs.shape[1] - 1

    torch.manual_seed(15033019)
    selfAttention_v1 = SelfAttentionv1(d_in, d_out)
    context_vector = selfAttention_v1(inputs)

    print("Context Vector\n", context_vector)
