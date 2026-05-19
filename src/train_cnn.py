import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import GTSRB_CNN
import time
import pandas as pd
import json

def train():
    data_dir = 'data/raw'
    epochs = 15
    batch_size = 64
    learning_rate = 0.001
    save_dir = 'models'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)
    
    model = GTSRB_CNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'gtsrb_cnn.pth'))
            print("  --> Saved Best Model")
            
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
    cnn_time = time.time() - start_time
    print(f"CNN Training completed in {cnn_time:.2f} seconds.")
    
    # Save history and times
    os.makedirs('reports', exist_ok=True)
    pd.DataFrame(history).to_csv('reports/training_history.csv', index=False)
    
    times_file = 'reports/training_times.json'
    times = {}
    if os.path.exists(times_file):
        with open(times_file, 'r') as f:
            times = json.load(f)
    times['CNN'] = cnn_time
    with open(times_file, 'w') as f:
        json.dump(times, f)

if __name__ == '__main__':
    train()
