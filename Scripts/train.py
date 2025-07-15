import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score, f1_score
from timesnet_model import TimesNet  # Custom TimesNet implementation

class Config:
    """Model and training configuration."""
    seq_len = 1024       # Input sequence length (EEG time steps)
    enc_in = 23          # Number of EEG channels
    d_model = 64         # Model embedding dimension
    top_k = 3            # TimesNet top-k frequencies
    e_layers = 3         # Number of encoder layers
    num_class = 2        # Binary classification (seizure vs non-seizure)

    batch_size = 32      # Training batch size
    epochs = 50          # Total training epochs
    learning_rate = 1e-4 # Optimizer learning rate
    val_split = 0.2      # Validation set ratio (20%)

def load_data(data_path, label_path):
    """Load and preprocess EEG data and labels."""
    X = np.load(data_path).astype(np.float32)  # EEG data [samples, channels, time]
    y = np.load(label_path).astype(np.int64)   # Labels (0 or 1)

    # Ensure channel dimension is correct (transpose if needed)
    if X.shape[1] != Config.enc_in:
        X = np.transpose(X, (0, 2, 1))  # [samples, T, C] → [samples, C, T]

    print(f"Data shape: {X.shape}, Labels shape: {y.shape}, dtype: {y.dtype}")
    return X, y

def create_datasets(X, y, val_split):
    """Split data into training and validation sets."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

def train_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X = X.permute(0, 2, 1)  # [B, C, T] → [B, T, C] (TimesNet expects time-first)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)  # Average loss

def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            X = X.permute(0, 2, 1)
            outputs = model(X)
            loss = criterion(outputs, y)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # Class 1 probabilities
            total_loss += loss.item()

    return {
        'loss': total_loss / len(loader),
        'auc': roc_auc_score(y_true, y_pred),  # AUC-ROC
        'f1': f1_score(y_true, np.round(y_pred))  # Binary F1
    }

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)

    # Load data
    data_path = "processed/eeg_data_20250714_121315.npy"
    label_path = "processed/eeg_labels_20250714_121315.npy"
    try:
        X, y = load_data(data_path, label_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create data loaders
    train_set, val_set = create_datasets(X, y, config.val_split)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    # Initialize model
    model = TimesNet(config).to(device)

    # Handle class imbalance (weighted loss)
    count_0 = (y == 0).sum()  # Non-seizure samples
    count_1 = (y == 1).sum()  # Seizure samples
    class_weights = torch.tensor([
        len(y) / (2 * count_0),  # Weight for class 0
        len(y) / (2 * count_1)   # Weight for class 1
    ], device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    best_auc = 0
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), "outputs/best_model.pth")

        print(f"Epoch {epoch+1:02d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"F1 Score: {val_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()