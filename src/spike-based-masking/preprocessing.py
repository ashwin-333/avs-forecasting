import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pickle
import matplotlib.pyplot as plt
import h5py

class PEDRoDataset(Dataset):
    def __init__(self, data_dir, device, split='train', transform=None, max_samples=None, timesteps=1, ):
        self.split = split
        self.transform = transform
        self.width = 350
        self.height = 260
        self.timesteps = timesteps
        self.device = device

        self.data_dir = os.path.join(data_dir, split)
        self.xml_dir = os.path.join(data_dir.replace('numpy', 'xml'), split)
        self.frame_files = [f for f in sorted(os.listdir(self.data_dir)) if f.endswith('.npy') and self._has_single_bbox(f)]

        if max_samples is not None:
            self.frame_files = self.frame_files[:max_samples]

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        # Load frame and preprocess
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.data_dir, frame_file)
        events = torch.from_numpy(np.load(frame_path)).to(self.device)
        binned_events = self._bin_events(events)


        frame = torch.zeros((self.timesteps, 2, self.height, self.width), dtype=torch.float32)
        for t, bin in enumerate(binned_events):
            x, y, p = bin[:, 1], bin[:, 2], bin[:, 3]
            frame[t, p, y, x] = 1  # This ignores repeated events
        frame = self._mask_frame(frame, 10, 0.15)

        # Load bounding box
        box_tensor = self._create_bbox(frame_file)

        # Convert to PyTorch tensors
        return frame, box_tensor
    
    def _has_single_bbox(self, frame_file):
        xml_filename = frame_file.replace('.npy', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return len(root.findall('object')) == 1

    def _bin_events(self, events):
        times = events[:, 0]
        frame_window = (times[-1] - times[0]) // self.timesteps
        
        window_edges = torch.arange(self.timesteps + 1, device=times.device) * frame_window + times[0]
        
        indices = torch.searchsorted(times, window_edges)
        
        bins = [events[indices[i]:indices[i + 1]] for i in range(self.timesteps)]
        
        return bins

    
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
    
    def _mask_frame(self, tensor: torch.Tensor, cell_size: int, threshold: float) -> torch.Tensor:
        tensor = tensor.to(self.device)
        B, C, H, W = tensor.shape

        assert H % cell_size == 0 and W % cell_size == 0, "H and W must be divisible by cell_size"
        grid_h, grid_w = H // cell_size, W // cell_size

        tensor_view = tensor.view(B, C, grid_h, cell_size, grid_w, cell_size)
        tensor_view = tensor_view.permute(0, 2, 4, 1, 3, 5).reshape(B, grid_h, grid_w, C, cell_size, cell_size)

        non_zero_counts = (tensor_view != 0).sum(dim=(3, 4, 5)) 

        total_elements = C * cell_size * cell_size

        non_zero_fraction = non_zero_counts / total_elements

        mask = (non_zero_fraction >= threshold).float()

        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mask = mask.repeat(1, 1, 1, C, cell_size, cell_size)

        filtered_tensor_view = tensor_view * mask

        filtered_tensor = filtered_tensor_view.view(B, grid_h, grid_w, C, cell_size, cell_size)
        filtered_tensor = filtered_tensor.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return filtered_tensor


def preprocess_train(timesteps, device):
    data_dir = os.path.join('PEDRo-dataset', 'numpy')
    train_dataset = PEDRoDataset(data_dir=data_dir, split='train', timesteps=timesteps, device=device)
    val_dataset = PEDRoDataset(data_dir=data_dir, split='val', timesteps=timesteps, device=device)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
    print("done")
    return (train_loader, val_loader)

def preprocess_test(timesteps):
    data_dir = os.path.join('PEDRo-dataset', 'numpy')
    test_dataset = PEDRoDataset(data_dir=data_dir, split='test', timesteps=timesteps)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
    return test_loader

def custom_collate_fn(batch):
    samples = [sample for sample, _ in batch]
    samples = torch.stack(samples, 1)
    targets = torch.stack([target for _, target in batch])
    return (samples, targets) 

