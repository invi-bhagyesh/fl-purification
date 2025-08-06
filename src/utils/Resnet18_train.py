import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import os


def train_resnet18(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.0001, config=None):
    """Enhanced ResNet18 training based on Resnet18_train.py"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, true_labels in progress_bar:
            images, labels = images.to(device), true_labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'Loss': f"{running_loss/total:.4f}",
                'Acc': f"{100.*accuracy_score(all_train_labels, all_train_preds):.2f}%"
            })
            
        train_loss = running_loss / total
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        val_total = 0
        
        with torch.no_grad():
            for images, true_labels in val_loader:
                images, labels = images.to(device), true_labels.squeeze().long().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                val_total += labels.size(0)
                
        val_loss /= val_total
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        
        # Update scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss:.4f} | Train F1: {train_f1:.4f} "
              f"| Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        print(f"Train - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Val   - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), './models/best_resnet18.pth')
            print(f'Saved best ResNet18 model (val_f1: {best_val_f1:.4f})')
    
    return model 
