import numpy as np
import torch
from torch.utils.data import DataLoader
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

# Function to preprocess a video and convert it to frames
def preprocess_video(idx):
    events = np.load(file_paths[idx])
    return bin_events_to_frames(events)

# Create a dataset from preprocessed videos
dataset = [preprocess_video(i) for i in range(len(file_paths))]
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load and print each batch's shape and sample data
for batch_idx, data_batch in enumerate(data_loader):
    print(f"Batch {batch_idx + 1}")
    print("Data shape:", data_batch[0].size())
    print("Sample data:", data_batch[0])
