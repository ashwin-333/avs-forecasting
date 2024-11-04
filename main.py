import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms
from model import Net
import torch.nn as nn
import torch.optim as optim

class PEDRoDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory (e.g., 'Data/numpy/numpy').
            split (str): One of 'train', 'test', or 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = os.path.join(data_dir, split)
        self.xml_dir = os.path.join(data_dir.replace('numpy', 'xml'), split)
        self.frame_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npy')])
        self.transform = transform

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        # Load the numpy frame
        frame_path = os.path.join(self.data_dir, self.frame_files[idx])
        frame = np.load(frame_path)

        # Convert the frame to a tensor
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Parse the corresponding XML file
        xml_filename = self.frame_files[idx].replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract bounding box information
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert boxes to a tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Apply any transformations
        if self.transform:
            frame = self.transform(frame)

        sample = (frame, boxes)
        return sample
if __name__ == '__main__':
    # Define a transform if needed (e.g., normalization)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the single channel data
    ])

    # Create a dataset and DataLoader
    data_dir = 'PEDRo-dataset/numpy'
    train_dataset = PEDRoDataset(data_dir=data_dir, split='train', transform=transform)
    print("Number of samples in the dataset:", len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    #train model

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