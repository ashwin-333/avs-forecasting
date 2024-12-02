import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import matplotlib.pyplot as plt
from preprocessing import PEDRoDataset, mask_frame

def visualize_bounding_boxes(images, pred_boxes, target_boxes):
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axs = [axs]

    for idx, (image, pred_box, target_box) in enumerate(zip(images, pred_boxes, target_boxes)):
        if image.shape[0] == 2:
            pos = image[0].cpu().numpy()
            neg = image[1].cpu().numpy()
            img = np.zeros((3, image.shape[1], image.shape[2]))
            img[0] = pos
            img[1] = neg
            img = np.clip(img, 0, 1)
            img = img.transpose(1, 2, 0)
            axs[idx].imshow(img)
        elif image.shape[0] == 1:
            img = image.cpu().numpy().squeeze(0)
            axs[idx].imshow(img, cmap="gray")
        else:
            img = image.cpu().numpy().mean(axis=0)
            axs[idx].imshow(img, cmap="gray")

        pred_rect = patches.Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            linewidth=2,
            edgecolor='b',
            facecolor='none',
            label="Predicted"
        )
        target_rect = patches.Rectangle(
            (target_box[0], target_box[1]),
            target_box[2] - target_box[0],
            target_box[3] - target_box[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none',
            label="Target"
        )
        axs[idx].add_patch(pred_rect)
        axs[idx].add_patch(target_rect)
        axs[idx].legend()
        axs[idx].axis("off")

    plt.tight_layout()
    plt.show()

def calculate_iou(pred_box, target_box):
    xA = np.maximum(pred_box[:, 0], target_box[:, 0])
    yA = np.maximum(pred_box[:, 1], target_box[:, 1])
    xB = np.minimum(pred_box[:, 2], target_box[:, 2])
    yB = np.minimum(pred_box[:, 3], target_box[:, 3])

    interWidth = np.maximum(0, xB - xA)
    interHeight = np.maximum(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = np.maximum(1, (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1]))
    boxBArea = np.maximum(1, (target_box[:, 2] - target_box[:, 0]) * (target_box[:, 3] - target_box[:, 1]))

    iou = interArea / (boxAArea + boxBArea - interArea)
        
    return iou


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss 
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch}")
    
    return epoch, loss

def test_masking(data_dir, grid_size=10, threshold=0.7):
    train_dataset = PEDRoDataset(data_dir=data_dir, split='train', pickle_file='train.pkl', timesteps=4)

    print("Applying mask_frame to the first 5 frames and plotting...")

    for idx in range(5):
        frame, _ = train_dataset[idx]
        timestep_frame = frame[0].numpy()

        masked_frame = mask_frame(timestep_frame, grid_size=grid_size, threshold=threshold)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        if timestep_frame.shape[0] == 3:
            plt.imshow(timestep_frame.transpose(1, 2, 0))
        else:
            plt.imshow(timestep_frame[0], cmap='gray')
        plt.title(f"Original Frame {idx}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        if masked_frame.shape[0] == 3:
            plt.imshow(masked_frame.transpose(1, 2, 0))
        else:
            plt.imshow(masked_frame[0], cmap='gray')
        plt.title(f"Masked Frame {idx}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()