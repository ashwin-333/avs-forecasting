import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pickle

class PEDRoDataset(Dataset):

    def _has_single_bbox(self, frame_file):
        xml_filename = frame_file.replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return len(root.findall('object')) == 1

    def __init__(self, data_dir, split='train', transform=None, pickle_file='pedro_data.pkl', max_samples=None):
        self.split = split
        self.pickle_file = pickle_file
        self.transform = transform

        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                data = pickle.load(f)
            self.frames = data[split]['frames']
            self.boxes = data[split]['boxes']
        else:
            self.data_dir = os.path.join(data_dir, split)
            self.xml_dir = os.path.join(data_dir.replace('numpy', 'xml'), split)
            self.frame_files = [f for f in sorted(os.listdir(self.data_dir)) if f.endswith('.npy') and self._has_single_bbox(f)]
            if max_samples is not None:
                self.frame_files = self.frame_files[:max_samples]
            self.frames = []
            self.boxes = []
            for i, frame_file in enumerate(self.frame_files):
                print("Frame ", i)
                frame_path = os.path.join(self.data_dir, frame_file)
                events = np.load(frame_path)
                frame = np.zeros((260, 346), dtype=np.float32)
                x, y, p = events[:, 2], events[:, 1], events[:, 3]
                frame[x, y] = p
                frame = torch.from_numpy(frame).unsqueeze(0)
                self.frames.append(frame)
                xml_filename = frame_file.replace('.npy', '.xml')
                xml_path = os.path.join(self.xml_dir, xml_filename)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                bndbox = root.find('object').find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                box = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
                self.boxes.append(box)
            all_data = {split: {'frames': self.frames, 'boxes': self.boxes}}
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(all_data, f)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.boxes[idx]

def preprocess():
    data_dir = 'PEDRo-dataset/numpy'
    train_dataset = PEDRoDataset(data_dir=data_dir, split='train', pickle_file='pedro_data.pkl')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for _ in train_loader:
        pass
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    return train_loader
