import numpy as np
import torch
from torch import nn
from snntorch import surrogate
import matplotlib.pyplot as plt
from torchvision.ops import distance_box_iou_loss
from model import Model
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def train_model(trainloader, valloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    spike_grad = surrogate.atan()
    beta = 0.5

    net = Model(beta=beta, spike_grad=spike_grad).to(device)
    net.apply(initialize_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 100

    train_loss_hist = []
    train_iou_hist = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0
        epoch_train_iou = 0
        for i, (data, targets) in enumerate(iter(trainloader)):
            batch_size = data.size(1)
            data = data.to(device) # (T x B x C x H x W)
            targets = targets.squeeze(1) 
            targets = targets.to(device) # B x 4

            net.train()
            out = net(data)

            optimizer.zero_grad()
            train_loss = torch.sum(distance_box_iou_loss(out, targets)) / batch_size
            train_loss.backward()
            optimizer.step()

            train_loss_hist.append(train_loss.item())
            epoch_train_loss += train_loss.item()

            pred_box = out.detach().cpu().numpy()
            actual_box = targets.detach().cpu().numpy()

            train_iou = np.sum(calculate_iou(pred_box, actual_box), axis=0) / batch_size
            train_iou_hist.append(train_iou.item())
            epoch_train_iou += train_iou

            if i < 5 and epoch == 0:
                image = data[:, 0].cpu()
                image = image[-1]
                visualize_bounding_boxes([image], [pred_box[0]], [actual_box[0]])

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {train_loss.item():.2f}")
            print("Sample predicted Bounding Box: ", pred_box[-1])
            print("Sample actual Bounding Box: ", actual_box[-1])
            print(f"Train IoU: {train_iou * 100:.2f}%")
            print()

        avg_train_loss = epoch_train_loss / len(trainloader)
        avg_train_iou = epoch_train_iou / len(trainloader)

        train_loss_hist.append(avg_train_loss)
        train_iou_hist.append(avg_train_iou)

    val_loss, val_iou = validate(net, valloader, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_iou:.4f}")

    epochs = list(range(1, num_epochs + 1))

    avg_epoch_train_loss = [np.mean(train_loss_hist[i * len(trainloader):(i + 1) * len(trainloader)]) for i in range(num_epochs)]
    avg_epoch_train_iou = [np.mean(train_iou_hist[i * len(trainloader):(i + 1) * len(trainloader)]) for i in range(num_epochs)]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_epoch_train_loss, label="Average train loss per Epoch")
    plt.plot(epochs, val_loss, label="Average train loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_epoch_train_iou, label="Average train IoU per Epoch")
    plt.plot(epochs, val_iou, label="Average val IoU per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.tight_layout()
    plt.show()

def validate(net, valloader, device):
    net.eval()
    val_loss = 0
    val_iou = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for data, targets in valloader:
            batch_size = data.shape(1)
            # Move inputs and targets to the appropriate device
            data, targets = data.to(device), targets.squeeze(1).to(device)

            # Forward pass
            out = net(data)
            
            # Calculate the loss
            loss = torch.sum(distance_box_iou_loss(out, targets)) / batch_size
            val_loss += loss.item()  # Accumulate the validation loss

            # Example metric calculation (adjust as needed)
            # For classification:
            pred_box = out.detach().cpu()
            actual_box = targets.detach().cpu()
            iou = np.sum(calculate_iou(pred_box, actual_box), axis=0) / batch_size
            val_iou += iou.item()

            total_samples += targets.size(0)
    
    # Calculate average loss and accuracy
    avg_val_loss = val_loss / len(valloader)
    avg_val_iou = val_iou / len(valloader)  # Example: accuracy for classification

    return avg_val_loss, avg_val_iou


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

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        #if m.bias is not None:
            #nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        #if m.bias is not None:
            #nn.init.constant_(m.bias, 0)

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