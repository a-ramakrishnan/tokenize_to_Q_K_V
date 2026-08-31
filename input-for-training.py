# Ramakrishnan A
# How to create the inputs for training
# File 4 in the Build LLMs from scratch series
# # Lots of pranams to Sebastian Raschka

# Step 1 -> Splitting into tokens & making the vocabulary
# Step 2 -> How to create training pairs

import tiktoken
import re

# Load the text
with open('book_of_wonders.txt', "r", encoding="utf-8") as f:
    _text = f.read()

# LLMs are single word predictors; In a 10 word sentence, if the first 4 words are used as input at
# the start, then the 5th word is output, words 2-5 are inputs then the 6th word is output and so
# on. Lots of lectures in youtube explain this.

# Self supervised or un supervised learning; No labelling is needed

# Load the tokenizer
# Using the token set of GPT2
tokenizer = tiktoken.get_encoding("gpt2")

# Encode the text
encode_text = tokenizer.encode(_text)

# Finding the number of words; Simple approach
words = re.findall(r"\b[\w']+\b", _text)
word_count = len(words)
print("Number of characters in the text", len(_text))
print("Number of words in the text", word_count)
print("Number of tokens", len(encode_text))

# Removing first 100 tokens from dataset
encoded_sample = encode_text[:50]
# How many tokens are included as part of the input; Model is trained
# to look at 4 tokens to predict the next token
context_size = 4
X = encoded_sample[:context_size]
y = encoded_sample[1:context_size+1]

print(f"X: {X}")
print(f"y:      {y}")
print('What does this mean; If input is X[0] then output is y[0]; If input is X[0] and X[1] then output is y[1]; If input is X[0], X[1] and X[2] then output is y[2] and so on')
print("Look at Stride below")


for j in range(1, context_size+1):
    context = encoded_sample[:j]
    desired = encoded_sample[j]

    print(context, "-->", desired)
    print(tokenizer.decode(context), "->", tokenizer.decode([desired]))

print("\nPrinting with text\n")

for i in range(len(encoded_sample) - context_size):
    context = encoded_sample[i:i + context_size]
    desired = encoded_sample[i + context_size]
    context_text = tokenizer.decode(context)
    desired_text = tokenizer.decode([desired])
    print(context, "-->", desired)
    print(context_text, "-->", desired_text)


# Implement a data loader using a Sliding Window approach
