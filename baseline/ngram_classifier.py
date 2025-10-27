import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


def load_data(filepath):
    """
    Load TSV data with tweet and label columns.
    
    Args:
        filepath: Path to TSV file
    Returns:
        texts: List of tweet texts
        labels: List of labels
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['tweet', 'label'])
    return df['tweet'].tolist(), df['label'].tolist()


def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, output_dir):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        dataset_name: Name of dataset (dev/test)
        output_dir: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred, labels=['NOT', 'OFF'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NOT', 'OFF'], 
                yticklabels=['NOT', 'OFF'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}_{dataset_name}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def train_and_evaluate(X_train, y_train, X_dev, y_dev, X_test, y_test, 
                      model_type='svm', output_dir='results/'):
    """
    Train model with TF-IDF features and evaluate on dev and test sets.
    
    Args:
        X_train, y_train: Training data
        X_dev, y_dev: Development data
        X_test, y_test: Test data
        model_type: 'svm' or 'nb'
        output_dir: Directory to save visualizations
    Returns:
        results: Dictionary with metrics
    """
    # TF-IDF vectorization with unigrams and bigrams
    print("\nVectorizing text with TF-IDF (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, 
                                lowercase=True, min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Select model
    if model_type == 'nb':
        model = MultinomialNB()
        model_name = 'Naive_Bayes'
        model_display = 'Naive Bayes'
    else:
        model = LinearSVC(random_state=42, max_iter=1000)
        model_name = 'SVM'
        model_display = 'SVM (Linear)'
    
    # Train
    print(f"\nTraining {model_display}...")
    model.fit(X_train_vec, y_train)
    
    # Predict
    y_dev_pred = model.predict(X_dev_vec)
    y_test_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    dev_f1 = f1_score(y_dev, y_dev_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    dev_acc = accuracy_score(y_dev, y_dev_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    dev_f1_off = f1_score(y_dev, y_dev_pred, pos_label='OFF')
    dev_f1_not = f1_score(y_dev, y_dev_pred, pos_label='NOT')
    test_f1_off = f1_score(y_test, y_test_pred, pos_label='OFF')
    test_f1_not = f1_score(y_test, y_test_pred, pos_label='NOT')
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{model_display} - Development Set Results:")
    print(f"{'='*60}")
    print(f"Accuracy: {dev_acc:.4f}")
    print(f"Macro F1: {dev_f1:.4f}")
    print(f"F1 (OFF): {dev_f1_off:.4f}")
    print(f"F1 (NOT): {dev_f1_not:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_dev, y_dev_pred, digits=4))
    
    print(f"\n{'='*60}")
    print(f"{model_display} - Test Set Results:")
    print(f"{'='*60}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")
    print(f"F1 (OFF): {test_f1_off:.4f}")
    print(f"F1 (NOT): {test_f1_not:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred, digits=4))
    
    # Plot confusion matrices
    print(f"\n{'='*60}")
    print("Generating confusion matrices...")
    print(f"{'='*60}")
    plot_confusion_matrix(y_dev, y_dev_pred, model_name, 'dev', output_dir)
    plot_confusion_matrix(y_test, y_test_pred, model_name, 'test', output_dir)
    
    return {
        'model': model_display,
        'preprocessing': 'raw',  # Will be updated later for other preprocessing variants
        'dev_accuracy': dev_acc,
        'dev_f1_macro': dev_f1,
        'dev_f1_off': dev_f1_off,
        'dev_f1_not': dev_f1_not,
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1,
        'test_f1_off': test_f1_off,
        'test_f1_not': test_f1_not
    }


def save_results(results, output_path):
    """Save results to CSV file."""
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate n-gram baseline models for offensive language detection."
    )
    parser.add_argument("--train", type=str, required=True, help="Path to training TSV file")
    parser.add_argument("--dev", type=str, required=True, help="Path to dev TSV file")
    parser.add_argument("--test", type=str, required=True, help="Path to test TSV file")
    parser.add_argument("--model", type=str, default='svm', choices=['svm', 'nb'], 
                        help="Model type: 'svm' or 'nb'")
    parser.add_argument("--output", type=str, default='results/', 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print("="*60)
    print("Loading data...")
    print("="*60)
    X_train, y_train = load_data(args.train)
    X_dev, y_dev = load_data(args.dev)
    X_test, y_test = load_data(args.test)
    
    print(f"Train size: {len(X_train)}")
    print(f"Dev size: {len(X_dev)}")
    print(f"Test size: {len(X_test)}")
    
    # Train and evaluate
    results = train_and_evaluate(X_train, y_train, X_dev, y_dev, X_test, y_test, 
                                args.model, args.output)
    
    # Save results
    output_file = os.path.join(args.output, f'baseline_{args.model}_raw_results.csv')
    save_results(results, output_file)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*60)
