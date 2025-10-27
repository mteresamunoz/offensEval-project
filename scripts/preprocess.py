import pandas as pd
import argparse
import re
import os


def preprocess_raw(text):
    """
    No preprocessing - return text as-is.
    
    Args:
        text: Input tweet text
    Returns:
        Original text
    """
    return text


def preprocess_clean(text):
    """
    Clean preprocessing: Remove URLs and @mentions only.
    
    Args:
        text: Input tweet text
    Returns:
        Cleaned text
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_aggressive(text):
    """
    Aggressive preprocessing: Remove URLs, @mentions, #hashtags, numbers, special chars.
    
    Args:
        text: Input tweet text
    Returns:
        Aggressively cleaned text
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\S+', '', text)
    # Remove hashtags (keep the word after #)
    text = re.sub(r'#(\S+)', r'\1', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    return text


def process_file(input_path, output_path, preprocess_func, strategy_name):
    """
    Process a TSV file with given preprocessing function.
    
    Args:
        input_path: Path to input TSV
        output_path: Path to output TSV
        preprocess_func: Preprocessing function to apply
        strategy_name: Name of strategy (for logging)
    """
    print(f"\nProcessing {input_path} with {strategy_name} strategy...")
    
    # Load data
    df = pd.read_csv(input_path, sep='\t', header=None, names=['tweet', 'label'])
    
    # Apply preprocessing
    df['tweet'] = df['tweet'].apply(preprocess_func)
    
    # Remove empty tweets (can happen with aggressive preprocessing)
    original_len = len(df)
    df = df[df['tweet'].str.strip() != '']
    removed = original_len - len(df)
    
    if removed > 0:
        print(f"  Warning: Removed {removed} empty tweets after preprocessing")
    
    # Save
    df.to_csv(output_path, sep='\t', header=False, index=False)
    print(f"  Saved to {output_path} ({len(df)} examples)")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Twitter data with different strategies for offensive language detection."
    )
    parser.add_argument("--input_dir", type=str, default="data/", 
                        help="Directory containing original TSV files")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed/", 
                        help="Directory to save preprocessed files")
    parser.add_argument("--strategy", type=str, default="all", 
                        choices=['raw', 'clean', 'aggressive', 'all'],
                        help="Preprocessing strategy to apply")
    
    args = parser.parse_args()
    
    # Create output directories
    strategies = {
        'raw': preprocess_raw,
        'clean': preprocess_clean,
        'aggressive': preprocess_aggressive
    }
    
    if args.strategy == 'all':
        strategies_to_run = strategies
    else:
        strategies_to_run = {args.strategy: strategies[args.strategy]}
    
    # Files to process
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    
    print("="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    for strategy_name, preprocess_func in strategies_to_run.items():
        strategy_dir = os.path.join(args.output_dir, strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)
        
        print(f"\n--- Strategy: {strategy_name.upper()} ---")
        
        for filename in files:
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(strategy_dir, filename)
            
            if os.path.exists(input_path):
                process_file(input_path, output_path, preprocess_func, strategy_name)
            else:
                print(f"  Warning: {input_path} not found, skipping...")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED")
    print("="*60)
    print(f"\nPreprocessed data saved to: {args.output_dir}")
    print("\nDirectory structure:")
    for strategy in strategies_to_run.keys():
        print(f"  {args.output_dir}{strategy}/")
        for filename in files:
            print(f"    - {filename}")


if __name__ == "__main__":
    main()
