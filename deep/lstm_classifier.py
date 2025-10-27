"""
BiLSTM Classifier for Offensive Language Detection
Supports GloVe and FastText embeddings with multiple preprocessing strategies.
"""

import pandas as pd
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TweetDataset(Dataset):
    """
    PyTorch Dataset for tweets with pre-trained embeddings.
    """
    def __init__(self, texts, labels, vocab, embedding_matrix, max_len=64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.embedding_matrix = embedding_matrix
        self.max_len = max_len
        self.label_map = {'NOT': 0, 'OFF': 1}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx].lower().split()[:self.max_len]
        label = self.label_map[self.labels[idx]]
        
        # Convert tokens to indices
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in text]
        
        # Pad sequence
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        
        return torch.LongTensor(indices), torch.LongTensor([label])


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for offensive language detection.
    """
    def __init__(self, embedding_matrix, hidden_size=128, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embedding layer (frozen pre-trained embeddings)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        # BiLSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, 
                           batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer (hidden_size * 2 because bidirectional)
        self.fc = nn.Linear(hidden_size * 2, 2)
    
    def forward(self, x):
        # x shape: (batch_size, max_len)
        embedded = self.embedding(x)  # (batch_size, max_len, embedding_dim)
        
        # LSTM output
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Take last hidden states from both directions
        hidden_fwd = hidden[-2]  # Forward direction
        hidden_bwd = hidden[-1]  # Backward direction
        hidden_concat = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        
        # Apply dropout and fully connected
        out = self.dropout(hidden_concat)
        out = self.fc(out)
        
        return out


def load_embeddings(embedding_path, embedding_type='glove'):
    """
    Load pre-trained embeddings (GloVe or FastText format).
    
    Args:
        embedding_path: Path to embedding file
        embedding_type: 'glove' or 'fasttext'
    Returns:
        Dictionary mapping words to vectors
    """
    print(f"\nLoading {embedding_type.upper()} embeddings from {embedding_path}...")
    embeddings = {}
    
    with open(embedding_path, 'r', encoding='utf-8') as f:
        # FastText has a header line with vocab_size and dimensions
        first_line = f.readline().strip().split()
        
        # Check if first line is header (FastText) or data (GloVe)
        if embedding_type == 'fasttext' and len(first_line) == 2:
            # Skip header
            pass
        else:
            # First line is data, process it
            word = first_line[0]
            vector = np.asarray(first_line[1:], dtype='float32')
            embeddings[word] = vector
        
        # Process remaining lines
        for line in f:
            values = line.strip().split()
            if len(values) < 10:  # Skip malformed lines
                continue
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
            except ValueError:
                continue
    
    print(f"Loaded {len(embeddings)} word vectors")
    return embeddings


def build_vocab_and_embeddings(texts, embeddings, embedding_dim=200):
    """
    Build vocabulary and embedding matrix from training data.
    
    Args:
        texts: List of tweet texts
        embeddings: Pre-trained embedding dictionary
        embedding_dim: Dimension of embeddings
    Returns:
        vocab: Word to index mapping
        embedding_matrix: Numpy array of embeddings
    """
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())
    
    # Build vocab with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for word, count in word_counts.most_common():
        if count >= 2:  # Minimum frequency threshold
            vocab[word] = idx
            idx += 1
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Build embedding matrix
    embedding_matrix = np.random.normal(0, 0.1, (len(vocab), embedding_dim))
    embedding_matrix[0] = np.zeros(embedding_dim)  # PAD vector
    
    found = 0
    for word, idx in vocab.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
            found += 1
    
    print(f"Found {found}/{len(vocab)} words in embeddings ({found/len(vocab)*100:.1f}%)")
    
    return vocab, embedding_matrix


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.squeeze().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


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
    
    filename = f'confusion_matrix_{model_name.lower().replace("-", "_")}_{dataset_name}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Train BiLSTM model for offensive language detection."
    )
    parser.add_argument("--train", type=str, required=True, help="Path to training TSV")
    parser.add_argument("--dev", type=str, required=True, help="Path to dev TSV")
    parser.add_argument("--test", type=str, required=True, help="Path to test TSV")
    parser.add_argument("--embeddings", type=str, required=True, 
                       help="Path to embeddings file")
    parser.add_argument("--embedding_type", type=str, default='glove',
                       choices=['glove', 'fasttext'],
                       help="Type of embeddings")
    parser.add_argument("--preprocessing", type=str, default='raw',
                       choices=['raw', 'clean', 'aggressive'],
                       help="Preprocessing strategy")
    parser.add_argument("--embedding_dim", type=int, default=200,
                       help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="LSTM hidden size")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Max epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--max_len", type=int, default=64,
                       help="Max sequence length")
    parser.add_argument("--output", type=str, default='results/',
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("BILSTM CLASSIFIER FOR OFFENSIVE LANGUAGE DETECTION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Embeddings: {args.embedding_type.upper()} ({args.embedding_dim}d)")
    print(f"Preprocessing: {args.preprocessing}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Dropout: {args.dropout}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_len}")
    
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
    
    # Load embeddings
    embeddings = load_embeddings(args.embeddings, args.embedding_type)
    
    # Build vocab and embedding matrix
    vocab, embedding_matrix = build_vocab_and_embeddings(
        train_df['tweet'].tolist(), embeddings, args.embedding_dim
    )
    
    # Create datasets
    train_dataset = TweetDataset(train_df['tweet'].tolist(), train_df['label'].tolist(),
                                 vocab, embedding_matrix, args.max_len)
    dev_dataset = TweetDataset(dev_df['tweet'].tolist(), dev_df['label'].tolist(),
                               vocab, embedding_matrix, args.max_len)
    test_dataset = TweetDataset(test_df['tweet'].tolist(), test_df['label'].tolist(),
                                vocab, embedding_matrix, args.max_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Compute class weights
    labels = train_df['label'].tolist()
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"\nClass weights: NOT={class_weights[0]:.3f}, OFF={class_weights[1]:.3f}")
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = BiLSTMClassifier(embedding_matrix, args.hidden_size, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training loop with early stopping
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_dev_f1 = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on dev
        dev_preds, dev_labels = evaluate(model, dev_loader, device)
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
            model_path = os.path.join(args.output, f'best_lstm_{args.embedding_type}_{args.preprocessing}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  → New best model saved (F1: {dev_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    model_path = os.path.join(args.output, f'best_lstm_{args.embedding_type}_{args.preprocessing}.pt')
    model.load_state_dict(torch.load(model_path))
    print(f"\nLoaded best model from {model_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Dev set
    dev_preds, dev_labels = evaluate(model, dev_loader, device)
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
    test_preds, test_labels = evaluate(model, test_loader, device)
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
    model_name = f"BiLSTM-{args.embedding_type.upper()}"
    plot_confusion_matrix(dev_labels, dev_preds, model_name, 'dev', args.output)
    plot_confusion_matrix(test_labels, test_preds, model_name, 'test', args.output)
    
    # Save results to CSV (compatible with baseline format)
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    results = {
        'model': model_name,
        'preprocessing': args.preprocessing,
        'test_f1_macro': test_f1_macro,
        'test_accuracy': test_acc,
        'test_f1_off': test_f1_off,
        'test_f1_not': test_f1_not
    }
    
    results_df = pd.DataFrame([results])
    output_file = os.path.join(args.output, f'lstm_{args.embedding_type}_{args.preprocessing}_results.csv')
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
