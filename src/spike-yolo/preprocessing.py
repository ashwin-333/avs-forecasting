import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pickle

class SpikeYOLODataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, pickle_file='spikeyolo_data.pkl', max_samples=None, timesteps=10):
        self.split = split
        self.pickle_file = pickle_file
        self.transform = transform
        self.width = 346
        self.height = 260
        self.timesteps = timesteps

        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                data = pickle.load(f)
            self.frames = data[split]['frames']
            self.targets = data[split]['targets']
        else:
            self.data_dir = os.path.join(data_dir, split)
            self.xml_dir = os.path.join(data_dir.replace('numpy', 'xml'), split)
            self.frame_files = [f for f in sorted(os.listdir(self.data_dir)) if f.endswith('.npy') and self._has_single_bbox(f)]

            if max_samples is not None:
                self.frame_files = self.frame_files[:max_samples]

            self.frames = []
            self.targets = []

            for i, frame_file in enumerate(self.frame_files):
                frame_path = os.path.join(self.data_dir, frame_file)
                events = np.load(frame_path)
                binned_events = self._bin_events(events)

                frame = np.zeros((self.timesteps, 2, self.height, self.width), dtype=np.float32)
                for t, bin in enumerate(binned_events):
                    x, y, p = bin[:, 1], bin[:, 2], bin[:, 3]
                    frame[t, p, y, x] = 1

                self.frames.append(torch.from_numpy(frame))

                target = self._create_yolo_target(frame_file)
                self.targets.append(target)

            all_data = {split: {'frames': self.frames, 'targets': self.targets}}
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(all_data, f)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.targets[idx]
    
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
    
    def _create_yolo_target(self, frame_file):
        xml_filename = frame_file.replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / self.width
        y_center = (ymin + ymax) / 2.0 / self.height
        box_width = (xmax - xmin) / self.width
        box_height = (ymax - ymin) / self.height

        return torch.tensor([[x_center, y_center, box_width, box_height]], dtype=torch.float32)


def preprocess(timesteps=10):
    data_dir = os.path.join('PEDRo-dataset', 'numpy')
    train_dataset = SpikeYOLODataset(data_dir=data_dir, split='train', pickle_file='train_spikeyolo.pkl', timesteps=timesteps)
    val_dataset = SpikeYOLODataset(data_dir=data_dir, split='val', pickle_file='val_spikeyolo.pkl', timesteps=timesteps)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, val_loader

def custom_collate_fn(batch):
    spike_frames = [frame for frame, _ in batch]
    spike_frames = torch.stack(spike_frames, 0)
    targets = torch.cat([target for _, target in batch], dim=0)
    return spike_frames, targets
