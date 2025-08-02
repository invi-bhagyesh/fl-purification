import torch.optim as optim
import torch.nn.functional as F

def train_autoencoder(model, train_loader, val_loader, num_epochs=100, reg_strength=1e-9):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            # Forward pass for bottleneck noise (as in your model's forward)
            output = model(images)
            loss = F.mse_loss(output, images) + model.get_l2_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader.dataset):.6f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                val_output = model(val_images)
                loss = F.mse_loss(val_output, val_images)
                val_loss += loss.item() * val_images.size(0)
        print(f"Validation Loss: {val_loss/len(val_loader.dataset):.6f}")

    
