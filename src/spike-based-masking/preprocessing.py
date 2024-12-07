import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pickle
import matplotlib.pyplot as plt
import h5py

class PEDRoDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, memmap_file='pedro_data.npy', max_samples=None, timesteps=1):
        self.split = split
        self.memmap_file = memmap_file
        self.transform = transform
        self.width = 350
        self.height = 350
        self.timesteps = timesteps

        # Preallocate memmap if file doesn't exist
        if not os.path.exists(self.memmap_file):
            self.data_dir = os.path.join(data_dir, split)
            self.xml_dir = os.path.join(data_dir.replace('numpy', 'xml'), split)
            self.frame_files = [f for f in sorted(os.listdir(self.data_dir)) if f.endswith('.npy') and self._has_single_bbox(f)]

            if max_samples is not None:
                self.frame_files = self.frame_files[:max_samples]

            num_samples = len(self.frame_files)
            # Shape: (num_samples, timesteps, 2, height, width)
            frames_shape = (num_samples, self.timesteps, 2, self.height, self.width)
            boxes_shape = (num_samples, 1, 4)

            # Create memmap arrays
            self.frames = np.memmap(self.memmap_file, dtype=np.float32, mode='w+', shape=frames_shape)
            self.boxes = np.memmap(self.memmap_file.replace('.npy', '_boxes.npy'), dtype=np.float32, mode='w+', shape=boxes_shape)

            for i, frame_file in enumerate(self.frame_files):
                print(f"Processing frame {i}/{num_samples}...")

                # Process frame
                frame_path = os.path.join(self.data_dir, frame_file)
                events = np.load(frame_path)
                binned_events = self._bin_events(events)

                frame = np.zeros((self.timesteps, 2, self.height, self.width), dtype=np.float32)
                for t, bin in enumerate(binned_events):
                    x, y, p = bin[:, 1], bin[:, 2], bin[:, 3]
                    frame[t, p, y, x] = 1  # Set event

                self.frames[i] = frame

                # Process bounding box
                box = self._create_bbox(frame_file)
                self.boxes[i] = box

            # Flush data to disk
            self.frames.flush()
            self.boxes.flush()
        else:
            # Load existing memmap
            self.frames = np.memmap(self.memmap_file, dtype=np.float32, mode='r')
            self.boxes = np.memmap(self.memmap_file.replace('.npy', '_boxes.npy'), dtype=np.float32, mode='r')

        self.length = self.frames.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        frame = torch.tensor(self.frames[idx], dtype=torch.float32)
        box = torch.tensor(self.boxes[idx], dtype=torch.float32)
        return frame, box
    
    def _has_single_bbox(self, frame_file):
        xml_filename = frame_file.replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return len(root.findall('object')) == 1
    
    def _bin_events(self, events):
        times = events[:, 0]
        bins = np.linspace(times[0], times[-1], self.timesteps + 1)
        indices = np.searchsorted(times, bins)
        return [events[indices[i]:indices[i + 1]] for i in range(self.timesteps)]
    
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
    train_dataset = PEDRoDataset(data_dir=data_dir, split='train', hdf5_file='train.h5', timesteps=timesteps)
    val_dataset = PEDRoDataset(data_dir=data_dir, split='val', hdf5_file='val.h5', timesteps=timesteps)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, num_workers = 4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    print("done")
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

