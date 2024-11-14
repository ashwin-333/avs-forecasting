import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms
import train as train

class PEDRoDataset(Dataset):

    def _has_single_bbox(self, frame_file):
        xml_filename = frame_file.replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # return True if exactly one bounding box is present
        return len(root.findall('object')) == 1
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory (e.g., 'Data/numpy/numpy').
            split (str): One of 'train', 'test', or 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = os.path.join(data_dir, split)
        self.xml_dir = os.path.join(data_dir.replace('numpy', 'xml'), split)

        self.frame_files = [f for f in sorted(os.listdir(self.data_dir)) if f.endswith('.npy') and self._has_single_bbox(f)]
        
        self.transform = transform
        self.width = 346 #346
        self.height = 260 #260

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_path = os.path.join(self.data_dir, self.frame_files[idx])
        events = np.load(frame_path)

        frame = torch.zeros((self.height, self.width), dtype=torch.float32)
        for event in events:
            frame[event[2]][event[1]] = event[3]
            #only records last events
        
        frame = torch.unsqueeze(frame, 0)

        # XML stuff
        xml_filename = self.frame_files[idx].replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bndbox = root.find('object').find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        
        return frame, boxes  
        

def preprocess():
    # Create a dataset and DataLoader
    data_dir = 'PEDRo-dataset/numpy'
    train_dataset = PEDRoDataset(data_dir=data_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    return train_loader