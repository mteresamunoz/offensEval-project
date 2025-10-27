import gensim.downloader as api
import os

print("Downloading FastText English embeddings...")
print("This may take several minutes (file size: ~1GB)")

# Download FastText trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset
# 300 dimensions, 1M words
fasttext_model = api.load('fasttext-wiki-news-subwords-300')

# Save in text format for compatibility with your LSTM script
output_path = 'embeddings/fasttext-wiki-news-300d.txt'
os.makedirs('embeddings', exist_ok=True)

print(f"Saving to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as f:
    for word in fasttext_model.index_to_key:
        vector = ' '.join(map(str, fasttext_model[word]))
        f.write(f"{word} {vector}\n")

print("FastText embeddings downloaded and saved successfully!")
print(f"Vocabulary size: {len(fasttext_model.index_to_key)}")
