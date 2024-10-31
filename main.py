import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

# Parameters
WIDTH, HEIGHT = 346, 260
FRAME_DURATION_MS = 40
data_folder = "./PEDRo-dataset/numpy/train"

# Load paths to the first 10 videos
file_paths = glob.glob(os.path.join(data_folder, "*.npy"))[:10]

# Function to bin events into frames
def bin_events_to_frames(events):
    timestamps = events[:, 0]
    min_time, max_time = timestamps.min(), timestamps.max()
    total_duration = max_time - min_time
    num_frames = int(np.ceil(total_duration / FRAME_DURATION_MS)) + 1
    frames = torch.zeros((HEIGHT, WIDTH, num_frames), dtype=torch.float32)
    
    for t, x, y, _ in events:
        frame_idx = int((t - min_time) // FRAME_DURATION_MS)
        frames[int(y), int(x), frame_idx] += 1

    return frames

# Custom Dataset for loading preprocessed videos
class VideoDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        events = np.load(self.file_paths[idx])
        frames = bin_events_to_frames(events)
        return frames

# Create the dataset and dataloader
dataset = VideoDataset(file_paths)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Process each batch in the DataLoader and print the shape
for batch_idx, video in enumerate(data_loader):
    print(f"Video {batch_idx + 1} - Shape: {video[0].size()}")
