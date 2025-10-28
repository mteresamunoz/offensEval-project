"""
Transformer Fine-tuning for Offensive Language Detection
Supports BERT, RoBERTa, and other Hugging Face models with error analysis.
Compatible with baseline/LSTM results format.
"""

import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    accuracy_score, 
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OffensiveLanguageDataset(Dataset):
    """PyTorch Dataset for transformer tokenization."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'NOT': 0, 'OFF': 1}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label_map[self.labels[idx]]
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions, labels, and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NOT', 'OFF'],
                yticklabels=['NOT', 'OFF'])
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{model_name.lower().replace("/", "_")}_{dataset_name}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def error_analysis(y_true, y_pred, texts, probabilities, output_dir, model_name, preprocessing):
    """
    Perform detailed error analysis.
    
    Args:
        y_true: True labels (numeric 0/1)
        y_pred: Predicted labels (numeric 0/1)
        texts: Original tweet texts (list)
        probabilities: Softmax probabilities (N, 2)
        output_dir: Output directory
        model_name: Model identifier (e.g., 'bert-base')
        preprocessing: Preprocessing strategy
    """
    labels_str = ['NOT', 'OFF']
    max_probs = probabilities.max(axis=1)
    
    # Create detailed dataframe
    errors_df = pd.DataFrame({
        'text': texts,
        'true_label': [labels_str[y] for y in y_true],
        'pred_label': [labels_str[y] for y in y_pred],
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
    print(f"ERROR ANALYSIS: {model_name} ({preprocessing})")
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
        f.write(f"ERROR ANALYSIS: {model_name} ({preprocessing})\n")
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


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune transformers for offensive language detection."
    )
    parser.add_argument("--train", type=str, required=True, help="Path to training TSV")
    parser.add_argument("--dev", type=str, required=True, help="Path to dev TSV")
    parser.add_argument("--test", type=str, required=True, help="Path to test TSV")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=['bert-base-uncased', 'roberta-base', 'vinai/bertweet-base'],
                       help="Hugging Face model identifier")
    parser.add_argument("--preprocessing", type=str, default='raw',
                       choices=['raw', 'clean', 'aggressive'],
                       help="Preprocessing strategy")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Max sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=4,
                       help="Max epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--output", type=str, default='results/',
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model display name
    model_display = args.model_name.split('/')[-1].upper()
    
    print("\n" + "="*60)
    print("TRANSFORMER FINE-TUNING FOR OFFENSIVE LANGUAGE DETECTION")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Display name: {model_display}")
    print(f"Preprocessing: {args.preprocessing}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max length: {args.max_length}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    train_df = pd.read_csv(args.train, sep='\t', header=None, names=['tweet', 'label'])
    dev_df = pd.read_csv(args.dev, sep='\t', header=None, names=['tweet', 'label'])
    test_df = pd.read_csv(args.test, sep='\t', header=None, names=['tweet', 'label'])
    
    print(f"Train: {len(train_df)} examples")
    print(f"Dev: {len(dev_df)} examples")
    print(f"Test: {len(test_df)} examples")
    print(f"Class distribution (train): {dict(train_df['label'].value_counts())}")
    
    # Load tokenizer and model
    print("\n" + "="*60)
    print("LOADING MODEL AND TOKENIZER")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    ).to(device)
    
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = OffensiveLanguageDataset(
        train_df['tweet'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        args.max_length
    )
    dev_dataset = OffensiveLanguageDataset(
        dev_df['tweet'].tolist(),
        dev_df['label'].tolist(),
        tokenizer,
        args.max_length
    )
    test_dataset = OffensiveLanguageDataset(
        test_df['tweet'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Training loop with early stopping
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_dev_f1 = 0
    patience = 2
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Evaluate on dev
        dev_preds, dev_labels, dev_probs = evaluate(model, dev_loader, device)
        dev_f1 = f1_score(dev_labels, dev_preds, average='macro')
        dev_acc = accuracy_score(dev_labels, dev_preds)
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Dev F1: {dev_f1:.4f} | "
              f"Dev Acc: {dev_acc:.4f}")
        
        # Early stopping
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            patience_counter = 0
            # Save best model
            model_path = os.path.join(
                args.output, 
                f'best_{model_display.lower()}_{args.preprocessing}.pt'
            )
            torch.save(model.state_dict(), model_path)
            print(f"  → New best model saved (F1: {dev_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    model_path = os.path.join(args.output, f'best_{model_display.lower()}_{args.preprocessing}.pt')
    model.load_state_dict(torch.load(model_path))
    print(f"\nLoaded best model from {model_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Dev set
    dev_preds, dev_labels, dev_probs = evaluate(model, dev_loader, device)
    dev_f1_macro = f1_score(dev_labels, dev_preds, average='macro')
    dev_acc = accuracy_score(dev_labels, dev_preds)
    dev_f1_off = f1_score(dev_labels, dev_preds, pos_label=1)
    dev_f1_not = f1_score(dev_labels, dev_preds, pos_label=0)
    
    print(f"\nDevelopment Set:")
    print(f"  Accuracy:  {dev_acc:.4f}")
    print(f"  Macro F1:  {dev_f1_macro:.4f}")
    print(f"  F1 (OFF):  {dev_f1_off:.4f}")
    print(f"  F1 (NOT):  {dev_f1_not:.4f}")
    print("\n" + classification_report(dev_labels, dev_preds, 
                                       target_names=['NOT', 'OFF'], digits=4))
    
    # Test set
    test_preds, test_labels, test_probs = evaluate(model, test_loader, device)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1_off = f1_score(test_labels, test_preds, pos_label=1)
    test_f1_not = f1_score(test_labels, test_preds, pos_label=0)
    
    print(f"\nTest Set:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Macro F1:  {test_f1_macro:.4f}")
    print(f"  F1 (OFF):  {test_f1_off:.4f}")
    print(f"  F1 (NOT):  {test_f1_not:.4f}")
    print("\n" + classification_report(test_labels, test_preds,
                                       target_names=['NOT', 'OFF'], digits=4))
    
    # Plot confusion matrices
    plot_confusion_matrix(dev_labels, dev_preds, model_display, 'dev', args.output)
    plot_confusion_matrix(test_labels, test_preds, model_display, 'test', args.output)
    
    # Error analysis
    error_analysis(
        test_labels, 
        test_preds, 
        test_df['tweet'].tolist(),
        test_probs,
        args.output,
        model_display.lower(),
        args.preprocessing
    )
    
    # Save results to CSV (compatible with baseline/LSTM format)
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    results = {
        'model': model_display,
        'preprocessing': args.preprocessing,
        'test_f1_macro': test_f1_macro,
        'test_accuracy': test_acc,
        'test_f1_off': test_f1_off,
        'test_f1_not': test_f1_not
    }
    
    results_df = pd.DataFrame([results])
    output_file = os.path.join(
        args.output, 
        f'transformer_{model_display.lower()}_{args.preprocessing}_results.csv'
    )
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Best dev F1:  {best_dev_f1:.4f}")
    print(f"Test F1:      {test_f1_macro:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")


if __name__ == "__main__":
    main()
