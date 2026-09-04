# Ramakrishnan A
# How to create the inputs for training
# File 4 in the Build LLMs from scratch series
# Lots of pranams to Sebastian Raschka

# Step 1 -> Splitting into tokens & making the vocabulary
# Step 2 -> How to create training pairs

import tiktoken
import re
import torch
from torch.utils.data import Dataset, DataLoader


# Implement a data loader using a Sliding Window approach
# With Pytorch Tensors; This is to handle optimization for processing

# Dataloader should contain two tensors
# Input -> Text the LLMs sees
# Target -> Prediction set

# Max_length -> The length of each set of input & output pair
# Stride -> The length by which the loader moves while creating the input & target sets
# For example, assume the tokens to be from 0 to 9
# If Stride is 4, then input will be [0,1,2,3], [4,5,6,7], [7,8,9] & output will be [1,2,3,4], [5,6,7,8] and [8,9]
# If Stride is 1, then inputs will be [0,1,2,3], [1,2,3,4], [2,3,4,5] and so on and targets will be [1,2,3,4], [2,3,4,5], [3,4,5,6] and so on
class GPTDataSet(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the text
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(tokens) - max_length, stride):
            input = tokens[i:max_length + i]
            target = tokens[i + 1:max_length + i + 1]
            self.input_ids.append(torch.tensor(input))
            self.target_ids.append(torch.tensor(target))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, ids):
        return self.input_ids[ids], self.target_ids[ids]


# Let's create the actual loader
# Arguments: input file
# Batch Size -> Batches processed by the system before model updates its parameters
# It directly trades off training stability vs. speed vs. memory
# Max Length -> Context Length (Number of words it can look at before predicting next
# word); sets how long a dependency the model is forced to learn in a single training
# example
# Stride -> Skip before next batch (input/output)
# num_workers -> Parallel processes
def createDataLoader(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataSet(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# Everything below only runs when this file is executed directly
# (python3 input_for_training.py) -- NOT when it's imported by another
# file like token_embeddings.py. This is what stops the demo prints,
# the book_of_wonders.txt read, and the dataloader iteration from
# firing off as a side effect of just importing createDataLoader.
if __name__ == "__main__":

    # LLMs are single word predictors; In a 10 word sentence, if the first 4 words are used as input at
    # the start, then the 5th word is output, words 2-5 are inputs then the 6th word is output and so
    # on. Lots of lectures in youtube explain this.

    # Self supervised or un supervised learning; No labelling is needed

    # Load the text
    with open('book_of_wonders.txt', "r", encoding="utf-8") as f:
        _text = f.read()

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
    y = encoded_sample[1:context_size + 1]

    print(f"X: {X}")
    print(f"y:      {y}")
    print('What does this mean; If input is X[0] then output is y[0]; If input is X[0] and X[1] then output is y[1]; If input is X[0], X[1] and X[2] then output is y[2] and so on')
    print("Look at Stride below")

    for j in range(1, context_size + 1):
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

    dataloader = createDataLoader(_text, batch_size=1, max_length=4, stride=1, shuffle=False)

    data = iter(dataloader)
    inputs, targets = next(data)

    input_ids = inputs[0].tolist()
    target_ids = targets[0].tolist()

    input_text = tokenizer.decode(input_ids)
    target_text = tokenizer.decode(target_ids)

    print("Input IDs:   ", input_ids)
    print("Input text:  ", input_text)
    print("Target IDs:  ", target_ids)
    print("Target text: ", target_text)

    print("First 3 batches")
    for _ in range(3):
        inputs, targets = next(data)
        print("Input IDs", inputs[0].tolist())
        print(tokenizer.decode(inputs[0].tolist()))
        print('Target IDs', targets[0].tolist())
        print(tokenizer.decode(targets[0].tolist()))

    print("""
        For some explanation on the params that get used in Data Loader/DataSet
        Batch Size -> Batches processed by the system before model updates its parameters It directly trades off training stability vs. speed vs. memory
        Max Length -> Context Length (Number of words it can look at before predicting next word); sets how long a dependency the model is forced to learn in a single training example
        Stride -> Skip before next batch (input/output)
        num_workers -> Parallel processes
        """)
