import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            spk3_rec, mem3_rec = model(inputs)
            
            # Since `spk3_rec` is a time series, we take the mean over time steps
            # before passing it to the loss (or the final time step if preferred)
            output = spk3_rec.mean(dim=0)  # Average spike activity over time steps

            # Calculate loss
            loss = criterion(output, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    print("Training complete.")