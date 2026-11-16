# Ramakrishnan A
# File 5-D in the series on Building LLMs from scratch
# One-time downloader and local cache setup for word2vec-google-news-300

import gensim.downloader as api

print("Downloading word2vec-google-news-300 (one-time, ~1.6GB)...")
model = api.load("word2vec-google-news-300")

save_path = "word2vec-google-news-300.kv"
model.save(save_path)
print(f"Saved to {save_path} -- future runs load from here, no download needed.")
