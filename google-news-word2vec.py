# Ramakrishnan A
# File 5-C in the series on Building LLMs from scratch
# Exploring semantic relationships in 300D Google News Word2Vec embeddings

import gensim.downloader as api
import numpy as np
model = api.load("word2vec-google-news-300")

word_vectors = model
#print(word_vectors['computer'])

print(word_vectors['dog'].shape)

print(word_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn = 5))
print(word_vectors.most_similar(positive=['Vehicles', 'Pollution'], negative=['Petrol'], topn = 5))
#print(word_vectors.most_similar(positive=['Mumbai', 'roads'], negative=['vehicles'], topn = 5))

print(word_vectors.similarity('dog', 'cat'))
print(word_vectors.similarity('Bengaluru', 'Mumbai'))
print(word_vectors.similarity('Shanghai', 'London'))
print(word_vectors.similarity('Bus', 'Book'))

print(word_vectors.most_similar("city", topn=10))

word1 = 'king'
word2 = 'monarch'

word3 = 'Bengaluru'
word4 = 'Avengers'

diff1 = model[word1] - model[word2]
diff3 = model[word3] - model[word4]

print("Calculating Vector Difference size")
size_diff1 = np.linalg.norm(diff1)
size_diff3 = np.linalg.norm(diff3)

print(f"Magnitude of difference between '{word1}' and '{word2}' is {size_diff1:.2f}")
print(f"Magnitude of difference between '{word3}' and '{word4}' is {size_diff3:.2f}")
