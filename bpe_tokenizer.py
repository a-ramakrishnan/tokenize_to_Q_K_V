# Ramakrishnan A
# File 2 in the series on Building LLMs from scratch
# Lots of pranams to Sebastian Raschka

# This is a BPE tokenizer using GPT-2's tiktoken vocabulary.
# PDF documents are handled in token_embeddings_with_pdf.py (File 5-B).

# Dataset -> Book of Wonders downloaded from Project Gutenberg

from importlib.metadata import version
import tiktoken
import re
import string

extra_unicode_punct = '’‘“”—–•…«»„'
pattern = r'([' + re.escape(string.punctuation + extra_unicode_punct) + r']| --|\s)'

with open('book_of_wonders.txt', "r", encoding="utf-8") as f:
    _text = f.read()

print("Total number of characters present:", len(_text))
print(_text[:299])
print("\n")

# Onto Tiktoken
print("Tiktoken Version:", version("tiktoken"))

# Using the token set of GPT2
tokenizer = tiktoken.get_encoding("gpt2")

print("Giving the tokenizer the first 200 odd words from the text")

ex_text = ("""
    The ProjectGutenbergeBook of The book of wonders

This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online
at www.gutenberg.org. If you are not located in the United States,
you will have to check the laws of the country where you are located
before using this eBook.

Title: The book of wonders
        gives plain and simple answers to the thousands of everyday questions that are asked and which all should be able to, but cannot answer...

Editor: Rudolph J. Bodmer



Release date: April 24, 2025 [eBook #75948]

Language: English

Original publication: New York: Presbrey Syndicate, Inc., 1914

Other information and formats: www.gutenberg.org/ebooks/75948""")

ids = tokenizer.encode(ex_text)

print(ids)
print(len(ids))

print("Converting back to text")
print(tokenizer.decode(ids))

print("To illustrate breaking down of words")
print(tokenizer.decode(ids[:12]))

print(tokenizer.decode([20336]))
