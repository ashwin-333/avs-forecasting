import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms
from train import train

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
        self.frame_files = self.frame_files[:100] #for now to make code run
        
        self.transform = transform
        self.width = 346
        self.height = 260
        

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_path = os.path.join(self.data_dir, self.frame_files[idx])
        events = np.load(frame_path)

        frame = torch.zeros((self.height, self.width), dtype=torch.int16)
        for event in events:
            frame[event[2]][event[1]] = event[3]
            #only records last events
        
        torch.unsqueeze(frame, 0)


        # XML stuff
        xml_filename = self.frame_files[idx].replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Bounding box
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            break #remove this when we need more than 1 bounding box
        
        boxes = torch.tensor(boxes, dtype=torch.int16)
    
        return frame, boxes  
        

if __name__ == '__main__':

    # Create a dataset and DataLoader
    data_dir = 'PEDRo-dataset/numpy'
    train_dataset = PEDRoDataset(data_dir=data_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)



    # Example usage
    for frames, boxes in train_loader:
        print("Frames shape:", frames[0])
        #frame dim: (num_frames, channels, height, width)
        print("Boxes:", boxes)
        break #just to print 1

    #train(train_dataset)
