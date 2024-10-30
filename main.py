import numpy as np
import torch
import os
import glob

WIDTH, HEIGHT = 346, 260
FRAME_DURATION_MS = 40

# only load paths to the first 10 videos
data_folder = "./PEDRo-dataset/numpy/train"
file_paths = glob.glob(os.path.join(data_folder, "*.npy"))[:10]

def bin_events_to_frames(events):
    timestamps = events[:, 0]
    min_time, max_time = timestamps.min(), timestamps.max()

    # calculating the total duration in milliseconds
    total_duration = max_time - min_time

    # calculating the number of frames
    num_frames = int(np.ceil(total_duration / FRAME_DURATION_MS)) + 1

    # initalizing the tensor for frames: (height, width, num_frames)
    frames = torch.zeros((HEIGHT, WIDTH, num_frames), dtype=torch.float32)

    # populating the frames with events
    for t, x, y, _ in events:
        frame_idx = int((t - min_time) // FRAME_DURATION_MS)
        frames[int(y), int(x), frame_idx] += 1

    return frames

def preprocess_video(file_path):
    events = np.load(file_path)
    return bin_events_to_frames(events)

# print the shapes of the 10 videos
for i, file_path in enumerate(file_paths):
    video = preprocess_video(file_path)
    print(f"Video {i + 1} - Shape: {video.shape}")
