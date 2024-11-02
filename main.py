import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms

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

        sample = {'frame': frame, 'boxes': boxes}
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

    # Example usage
    for batch in train_loader:
        frames = batch['frame']  # Shape: [batch_size, 1, height, width]
        boxes = batch['boxes']   # Shape: [batch_size, num_boxes, 4]
        print("Frames shape:", frames.shape)
        print("Boxes:", boxes)