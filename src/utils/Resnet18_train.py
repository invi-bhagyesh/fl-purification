NUM_CLASSES = 8
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.0001
#1e-5 seems best
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

from sklearn.metrics import f1_score

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.squeeze().long().to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        all_train_preds.extend(predicted.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())
        total += labels.size(0)
    train_loss = running_loss / total
    train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

    # Validation
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.squeeze().long().to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
            val_total += labels.size(0)
    val_loss /= val_total
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

    print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {train_loss:.4f} | Train F1: {train_f1:.4f} "
          f"| Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")


