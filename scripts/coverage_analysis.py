# save as: scripts/coverage_analysis.py

import pandas as pd
import os

def load_embedding_vocab(embedding_path):
    """Load vocabulary from embedding file"""
    vocab = set()
    print(f"Loading vocabulary from {embedding_path}...")
    
    with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"  Processed {i} words...")
            
            parts = line.strip().split()
            if len(parts) > 1:
                word = parts[0]
                vocab.add(word)
    
    print(f"  Total vocabulary size: {len(vocab)}")
    return vocab

def analyze_all_strategies():
    """Calculate coverage for both GloVe and FastText across all preprocessing strategies"""
    
    # Define embedding paths
    glove_path = 'embeddings/glove.twitter.27B.200d.txt'
    fasttext_path = 'embeddings/fasttext-wiki-news-300d.txt'
    
    # Load vocabularies
    print("\n" + "="*70)
    print("LOADING EMBEDDINGS")
    print("="*70)
    
    glove_vocab = load_embedding_vocab(glove_path)
    fasttext_vocab = load_embedding_vocab(fasttext_path)
    
    # Preprocessing strategies
    strategies = ['raw', 'clean', 'aggressive']
    results = []
    
    print("\n" + "="*70)
    print("CALCULATING COVERAGE")
    print("="*70)
    
    for strategy in strategies:
        print(f"\nProcessing {strategy} preprocessing...")
        
        # Load preprocessed data
        data_path = f'data/preprocessed/{strategy}/train.tsv'
        df = pd.read_csv(data_path, sep='\t', header=None, quoting=3)
        tweets = df[0].tolist()
        
        print(f"  Total tweets: {len(tweets)}")
        
        # Tokenize (simple whitespace split)
        tokens = []
        for tweet in tweets:
            tokens.extend(tweet.split())
        
        unique_tokens = set(tokens)
        vocab_size = len(unique_tokens)
        
        print(f"  Unique tokens: {vocab_size}")
        
        # Calculate GloVe coverage
        glove_covered = len(unique_tokens & glove_vocab)
        glove_coverage = (glove_covered / vocab_size) * 100
        
        print(f"  GloVe coverage: {glove_coverage:.2f}%")
        
        # Calculate FastText coverage
        fasttext_covered = len(unique_tokens & fasttext_vocab)
        fasttext_coverage = (fasttext_covered / vocab_size) * 100
        
        print(f"  FastText coverage: {fasttext_coverage:.2f}%")
        
        results.append({
            'Preprocessing': strategy,
            'Vocab Size': vocab_size,
            'GloVe Coverage (%)': round(glove_coverage, 2),
            'FastText Coverage (%)': round(fasttext_coverage, 2)
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    results_df = analyze_all_strategies()
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(results_df)
    results_df.to_csv('results/embedding_coverage.csv', index=False)
    print(f"\nResults saved to results/embedding_coverage.csv")
