import pandas as pd
import numpy as np
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
        DataFrame with tweets and labels
    """
    df = pd.read_csv(filepath, sep='\t', header=None, names=['tweet', 'label'])
    return df



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



def error_analysis_baseline(y_true, y_pred, texts, probabilities, output_dir, model_name, preprocessing):
    """
    Perform error analysis for baseline models (SVM, Naive Bayes).
    
    Args:
        y_true: True labels (strings 'NOT'/'OFF')
        y_pred: Predicted labels (strings 'NOT'/'OFF')
        texts: Tweet texts
        probabilities: Prediction probabilities (N, 2) array
        output_dir: Output directory
        model_name: Model name (e.g., 'svm', 'naive_bayes')
        preprocessing: Preprocessing strategy
    """
    
    # Get max confidence
    max_probs = probabilities.max(axis=1)
    
    # Create detailed dataframe
    errors_df = pd.DataFrame({
        'text': texts,
        'true_label': y_true,
        'pred_label': y_pred,
        'confidence': max_probs,
        'correct': y_true == y_pred,
        'text_length': [len(str(t).split()) for t in texts]
    })
    
    # Save all predictions
    pred_file = f'{output_dir}/predictions_{model_name}_{preprocessing}.csv'
    errors_df.to_csv(pred_file, index=False)
    
    # Analyze errors
    errors = errors_df[~errors_df['correct']]
    
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {model_name.upper()} ({preprocessing})")
    print(f"{'='*60}")
    
    print(f"\nTotal errors: {len(errors)}/{len(errors_df)} ({len(errors)/len(errors_df)*100:.1f}%)")
    
    # Confusion breakdown
    off_as_not = len(errors[(errors['true_label']=='OFF') & (errors['pred_label']=='NOT')])
    not_as_off = len(errors[(errors['true_label']=='NOT') & (errors['pred_label']=='OFF')])
    
    print(f"\nConfusion patterns:")
    print(f"  OFF → NOT (false negatives): {off_as_not} ({off_as_not/len(errors)*100:.1f}%)")
    print(f"  NOT → OFF (false positives): {not_as_off} ({not_as_off/len(errors)*100:.1f}%)")
    
    # High-confidence errors
    high_conf_errors = errors[errors['confidence'] > 0.8].sort_values('confidence', ascending=False)
    print(f"\nHigh-confidence errors (conf > 0.8): {len(high_conf_errors)}")
    
    # Error by length
    errors_df['length_bin'] = pd.cut(
        errors_df['text_length'], 
        bins=[0, 5, 10, 20, 100], 
        labels=['Very short (≤5)', 'Short (6-10)', 'Medium (11-20)', 'Long (>20)']
    )
    
    print(f"\nError rate by text length:")
    for length_bin in ['Very short (≤5)', 'Short (6-10)', 'Medium (11-20)', 'Long (>20)']:
        group = errors_df[errors_df['length_bin'] == length_bin]
        if len(group) > 0:
            error_rate = (1 - group['correct'].mean()) * 100
            print(f"  {length_bin}: {error_rate:.1f}% ({len(group)} tweets)")
    
    # Save detailed report
    report_file = f'{output_dir}/error_analysis_{model_name}_{preprocessing}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"ERROR ANALYSIS: {model_name.upper()} ({preprocessing})\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total errors: {len(errors)}/{len(errors_df)} ({len(errors)/len(errors_df)*100:.1f}%)\n\n")
        f.write(f"Confusion:\n")
        f.write(f"  OFF → NOT (false negatives): {off_as_not}\n")
        f.write(f"  NOT → OFF (false positives): {not_as_off}\n\n")
        
        if len(high_conf_errors) > 0:
            f.write("Top 20 high-confidence errors:\n")
            f.write("-"*60 + "\n")
            for idx, row in high_conf_errors.head(20).iterrows():
                f.write(f"\n[{row['true_label']}→{row['pred_label']}, conf={row['confidence']:.3f}]\n")
                f.write(f"{row['text']}\n")
    
    print(f"Detailed error analysis saved to: {report_file}")



def train_and_evaluate(train_df, dev_df, test_df,
                      model_type='svm', preprocessing='raw', output_dir='results/'):
    """
    Train model with TF-IDF features and evaluate on dev and test sets.
    
    Args:
        train_df, dev_df, test_df: DataFrames with tweet and label columns
        model_type: 'svm' or 'nb'
        preprocessing: 'raw', 'clean', or 'aggressive'
        output_dir: Directory to save visualizations
    Returns:
        results: Dictionary with metrics
    """
    X_train, y_train = train_df['tweet'].tolist(), train_df['label'].tolist()
    X_dev, y_dev = dev_df['tweet'].tolist(), dev_df['label'].tolist()
    X_test, y_test = test_df['tweet'].tolist(), test_df['label'].tolist()
    
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
    
    # Predict on dev
    y_dev_pred = model.predict(X_dev_vec)
    
    # Predict on test with probabilities for error analysis
    y_test_pred = model.predict(X_test_vec)
    
    # Get probabilities (need to handle SVM differently)
    if model_type == 'nb':
        y_test_proba = model.predict_proba(X_test_vec)
    else:
        # For SVM, use decision function and convert to pseudo-probabilities
        decision = model.decision_function(X_test_vec)
        # Normalize to [0, 1] range as pseudo-probabilities
        from scipy.special import expit
        proba_off = expit(decision)  # Sigmoid for OFF class
        proba_not = 1 - proba_off    # NOT class
        y_test_proba = np.column_stack([proba_not, proba_off])
    
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
    
    # ERROR ANALYSIS
    print(f"\n{'='*60}")
    print("PERFORMING ERROR ANALYSIS")
    print(f"{'='*60}")
    error_analysis_baseline(
        y_true=y_test,
        y_pred=y_test_pred,
        texts=X_test,
        probabilities=y_test_proba,
        output_dir=output_dir,
        model_name=model_name.lower(),
        preprocessing=preprocessing
    )
    
    return {
        'model': model_display,
        'preprocessing': preprocessing,
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
    parser.add_argument("--preprocessing", type=str, default='raw', 
                        choices=['raw', 'clean', 'aggressive'],
                        help="Preprocessing strategy used")
    parser.add_argument("--output", type=str, default='results/', 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    train_df = load_data(args.train)
    dev_df = load_data(args.dev)
    test_df = load_data(args.test)
    
    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Preprocessing: {args.preprocessing}")
    print(f"Model: {args.model.upper()}")
    
    # Train and evaluate
    results = train_and_evaluate(
        train_df, dev_df, test_df,
        args.model, args.preprocessing, args.output
    )
    
    # Save results
    output_file = os.path.join(args.output, f'baseline_{args.model}_{args.preprocessing}_results.csv')
    save_results(results, output_file)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*60)
