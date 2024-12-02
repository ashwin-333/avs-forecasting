import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pickle
import matplotlib.pyplot as plt

class PEDRoDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, pickle_file='pedro_data.pkl', max_samples=None, timesteps=1):
        self.split = split
        self.pickle_file = pickle_file
        self.transform = transform
        #adding padding to images
        self.width = 350
        self.height = 350
        self.timesteps = timesteps

        #loads pkl files
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                data = pickle.load(f)
            self.frames = data[split]['frames']
            self.boxes = data[split]['boxes']

        #creates pkl file
        else: 
            self.data_dir = os.path.join(data_dir, split)
            self.xml_dir = os.path.join(data_dir.replace('numpy', 'xml'), split)
            self.frame_files = [f for f in sorted(os.listdir(self.data_dir)) if f.endswith('.npy') and self._has_single_bbox(f)]

            #limit frames for debugging
            if max_samples is not None:
                self.frame_files = self.frame_files[:max_samples]

            self.frames = []
            self.boxes = []
            #shape = (len(self.frame_files), 2, self.height, self.width) # Frames x C x H x W
            #data_array = np.memmap('pickle_file', dtype=np.float32, mode='w+', shape=shape)

            for i, frame_file in enumerate(self.frame_files):
                print("Frame ", i)

                #frame
                frame_path = os.path.join(self.data_dir, frame_file)
                events = np.load(frame_path)
                binned_events = self._bin_events(events)

                frame = np.zeros((self.timesteps, 2, self.height, self.width), dtype=np.float32) #0 = negative events, 1 = positive events
                for t, bin in enumerate(binned_events):
                    x, y, p = bin[:, 1], bin[:, 2], bin[:, 3]
                    frame[t,p,y,x] = 1 #This ignores repeated events
                    #np.add.at(frame, (t, p, y, x), 1) #Adds up repeated events
                self.frames.append(torch.from_numpy(frame))

                #bounding box
                box = self._create_bbox(frame_file)
                self.boxes.append(box)

            all_data = {split: {'frames': self.frames, 'boxes': self.boxes}}
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(all_data, f)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.boxes[idx]
    
    def _has_single_bbox(self, frame_file):
        xml_filename = frame_file.replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return len(root.findall('object')) == 1
    
    def _bin_events(self, events):
        times = events[:, 0]
        frame_window = (times[-1] - times[0]) // self.timesteps
        window_start = np.arange(self.timesteps) * frame_window + times[0]
        window_end = window_start + frame_window
        indices_start = np.searchsorted(times, window_start)
        indices_end = np.searchsorted(times, window_end)
        slices = list(zip(indices_start, indices_end))

        return [events[start:end] for start, end in slices]
    
    def _create_bbox(self, frame_file):
        xml_filename = frame_file.replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        return torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)


def preprocess_train(timesteps):
    data_dir = os.path.join('PEDRo-dataset', 'numpy')
    train_dataset = PEDRoDataset(data_dir=data_dir, split='train', pickle_file='train.pkl', timesteps=timesteps)
    val_dataset = PEDRoDataset(data_dir=data_dir, split='val', pickle_file='val.pkl', timesteps=timesteps)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers = 4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    return (train_loader, val_loader)

def preprocess_test(timesteps):
    data_dir = os.path.join('PEDRo-dataset', 'numpy')
    test_dataset = PEDRoDataset(data_dir=data_dir, split='test', pickle_file='test.pkl', timesteps=timesteps)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    return test_loader

def custom_collate_fn(batch):
    samples = [sample for sample, _ in batch]
    samples = torch.stack(samples, 1)
    targets = torch.stack([target for _, target in batch])
    return (samples, targets) 

def mask_frame(frame, grid_size, threshold=0.7, width=None, height=None):
    _, h, w = frame.shape
    cell_size = grid_size * grid_size

    if h % grid_size != 0 or w % grid_size != 0:
        raise ValueError("Frame dimensions must be divisible by the grid size.")

    reshaped = frame[0].reshape(h // grid_size, grid_size, w // grid_size, grid_size)
    grid_true_ratios = reshaped.sum(axis=(1, 3)) / cell_size
    mask_grid = grid_true_ratios >= threshold
    mask = mask_grid.repeat(grid_size, axis=0).repeat(grid_size, axis=1)
    processed_frame = frame * mask[np.newaxis, :, :]

    return processed_frame

