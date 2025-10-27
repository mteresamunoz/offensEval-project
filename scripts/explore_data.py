import pandas as pd
import random
from collections import Counter
import argparse


def analyze_file(path, outfile):
    """
    Analyze a TSV dataset for basic statistics and save results to a text file.

    Args:
        path (str): Path to the input TSV file.
        outfile (str): Path to save the output statistics.
    """
    # Read TSV with no header, only 2 columns: tweet and label
    df = pd.read_csv(path, sep='\t', header=None, names=['tweet', 'label'])
    
    total = len(df)
    counts = Counter(df['label'])
    off_prop = counts.get('OFF', 0) / total
    not_prop = counts.get('NOT', 0) / total
    off_example = df[df['label']=='OFF']['tweet'].sample(1).values[0] if counts.get('OFF', 0) > 0 else "No OFF examples"
    not_example = df[df['label']=='NOT']['tweet'].sample(1).values[0] if counts.get('NOT', 0) > 0 else "No NOT examples"
    avg_len = round(df['tweet'].str.len().mean(), 2)

    # Print to console
    print(f"\n{'='*50}")
    print(f"File: {path}")
    print(f"{'='*50}")
    print(f"Number of examples: {total}")
    print(f"Class distribution: {dict(counts)}")
    print(f"Proportion OFF: {off_prop:.4f} ({counts.get('OFF', 0)} examples)")
    print(f"Proportion NOT: {not_prop:.4f} ({counts.get('NOT', 0)} examples)")
    print(f"\nSample OFF tweet:\n  {off_example}")
    print(f"\nSample NOT tweet:\n  {not_example}")
    print(f"\nAverage tweet length: {avg_len} characters")
    print(f"{'='*50}\n")

    # Save to file
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"File: {path}\n")
        f.write(f"Number of examples: {total}\n")
        f.write(f"Class distribution: {dict(counts)}\n")
        f.write(f"Proportion OFF: {off_prop:.4f} ({counts.get('OFF', 0)} examples)\n")
        f.write(f"Proportion NOT: {not_prop:.4f} ({counts.get('NOT', 0)} examples)\n")
        f.write(f"Sample OFF: {off_example}\n")
        f.write(f"Sample NOT: {not_example}\n")
        f.write(f"Average tweet length: {avg_len} characters\n")
        f.write("-" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore an OLID (tweets) TSV dataset and output basic statistics to a text file."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the input TSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output stats text file.")
    args = parser.parse_args()
    analyze_file(args.input, args.output)
