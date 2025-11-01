# save as: scripts/coverage_analysis.py

import pandas as pd
import numpy as np
from collections import Counter

def calculate_coverage(tokens_file, embedding_vocab):
    """
    Calculate embedding coverage for a tokenized dataset
    
    Args:
        tokens_file: List of all tokens in the dataset
        embedding_vocab: Set of words in the pre-trained embeddings
    
    Returns:
        coverage_pct: Percentage of tokens found in embeddings
        vocab_size: Size of unique vocabulary in dataset
    """
    unique_tokens = set(tokens_file)
    vocab_size = len(unique_tokens)
    
    covered = len(unique_tokens & embedding_vocab)
    coverage_pct = (covered / vocab_size) * 100
    
    return coverage_pct, vocab_size

# Example: Test with your preprocessed data
def analyze_all_strategies():
    # Load GloVe vocabulary
    glove_vocab = set()
    with open('embeddings/glove.twitter.27B.200d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.split()[0]
            glove_vocab.add(word)
    
    strategies = ['raw', 'clean', 'aggressive']
    results = []
    
    for strategy in strategies:
        # Load preprocessed data
        df = pd.read_csv(f'data/preprocessed/{strategy}/train.tsv', sep='\t', header=None)
        tweets = df[0].tolist()
        
        # Tokenize
        tokens = []
        for tweet in tweets:
            tokens.extend(tweet.split())
        
        # Calculate coverage
        coverage, vocab_size = calculate_coverage(tokens, glove_vocab)
        results.append({
            'Preprocessing': strategy,
            'Vocab Size': vocab_size,
            'Coverage (%)': coverage
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    results_df = analyze_all_strategies()
    print(results_df)
    results_df.to_csv('results/embedding_coverage.csv', index=False)
