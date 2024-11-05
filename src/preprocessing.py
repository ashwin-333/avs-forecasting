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
        self.height = 346
        self.width = 260

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        # Load the numpy frame
        frame_path = os.path.join(self.data_dir, self.frame_files[idx])
        frame = np.load(frame_path)  # Assuming frame shape is [N, 4] (event-based data)


        frame = frame[:100]
        # Convert events to a 2D frame of fixed size (346 x 260)
        frame_2d = torch.zeros((self.height, self.width), dtype=torch.float32)
        
        # Map events to pixels in the frame
        x_coords = torch.clamp(torch.tensor(frame[:, 0], dtype=torch.long), 0, self.width - 1)
        y_coords = torch.clamp(torch.tensor(frame[:, 1], dtype=torch.long), 0, self.height - 1)
        
        # Accumulate event counts at each pixel
        for x, y in zip(x_coords, y_coords):
            frame_2d[y, x] += 1
        
        # Add a channel dimension to match the expected input shape
        frame_2d = frame_2d.unsqueeze(0)

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
            frame_2d = self.transform(frame_2d)

        sample = {'frame': frame_2d, 'boxes': boxes}
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
