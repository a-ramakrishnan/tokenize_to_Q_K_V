# Ramakrishnan A
# File 1 in the series on Building LLMs from scratch
# Lots of pranams to Sebastian Raschka

# This is a regex-based tokenizer for text files. (Handles <|unk|> and <|endoftext|>)
# See bpe_tokenizer.py (File 2) for Byte-Pair Encoding (BPE), and
# token_embeddings_with_pdf.py (File 5-B) for PDF document ingestion.
#
# Dataset -> Book of Wonders downloaded from Project Gutenberg

import re
import string

extra_unicode_punct = '’‘“”—–•…«»„'
pattern = r'([' + re.escape(string.punctuation + extra_unicode_punct) + r']| --|\s)'

with open('book_of_wonders.txt', "r", encoding="utf-8") as f:
    _text = f.read()

print("Total number of characters present:", len(_text))
print(_text[:299])
print("\n")

# Now onto regular expressions for handling text splitting

ex_text = "Hi, Testing the first version. Let's see what happens!"
# Just on spaces and nothing else
result = re.split(r'(\s)', ex_text)
print(result)

# Splitting full stop and commas
result = re.split(r'([.,]|\s)', ex_text)
print(result)

# Handling redundant characters
# Note that this will also remove whitespaces; Not a good idea for coding examples
result = [item for item in result if item.strip()]
print(result)

# Handling more characters more cleanly
result = re.split(pattern, ex_text)
print(result)

# Taking out redundant words & characters
result = [item for item in re.split(pattern, ex_text) if item.strip()]
print(result)
print("----------End of Steps showing how a tokenizer is contructed------------\n")

# Now, let's move to the full text
processed = [item for item in re.split(pattern, _text) if item.strip()]
print("First 20 tokens", processed[:20])
print("Total number of tokens", len(processed))

print("\nLet's create Token IDs now")
print("First creating the vocabulary with unique tokens & IDs")

words = sorted(set(processed))
print("Length of Sorted set", len(words))
print("First 10 words in Sorted set", words[:9])
print("Last 10 words in Sorted set", words[-9:])

print("Onto the Vocabulary")
vocab = {token:integer for integer, token in enumerate(words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 10:
        break

class TokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {item:s for s, item in vocab.items() }

    def encode(self, text):
        processed = [item for item in re.split(pattern, text) if item.strip()]
        processed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in processed
        ]
        ids = [self.str_to_int[s] for s in processed]

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!;:"\'\(\)\{\}\[\]—–])', r'\1', text)
        return text

print("Instantiating the Class")
tokenizer = TokenizerV1(vocab)
sample_text ="Just Slashers are available here"
ids = tokenizer.encode(sample_text)
print("Sample Text -> ", sample_text)
print("Token IDs", ids)
print("Decoded Sample Text -> ", tokenizer.decode(ids))

print("\nSpecial Context Tokens")
print("Adding Unk and end of text tokens to the vocabulary")

tokens = sorted(list(set(processed)))
tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:integer for integer, token in enumerate(tokens)}
print("Length of new Vocabulary", len(vocab))
tokenizer = TokenizerV1(vocab)

for i, item in enumerate(list(vocab.items())[-10:]):
    print(item)

print("Encoding the full text now")
ids = tokenizer.encode(_text)
print("Last 10 Ids are ", ids[-10:])
print("Last 10 words are ", tokenizer.decode(ids[-10:]))


ids = tokenizer.encode(ex_text)
print("Last 10 Ids are ", ids[-10:])
print("Last 10 words are ", tokenizer.decode(ids[-10:]))
