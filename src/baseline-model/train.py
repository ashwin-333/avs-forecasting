import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms
from torch import nn
from snntorch import functional as SF
from snntorch import surrogate
import snntorch as snn
from snntorch import utils
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FIL(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, beta, spike_grad):
        super(FIL, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(mid_channels, out_channels),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism="none"),
        )
    def forward(self, x: torch.Tensor):
        x = torch.sum(x, dim=0)
        _, mem_rec = self.net(x)
        return mem_rec

class SCNN(nn.Module):
    def __init__(self, beta, spike_grad):
        super(SCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(128, 128, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(256, 256, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(512, 512, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(512, 512, 3),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x: torch.Tensor):
        spk_rec = []
        utils.reset(self.net)
        for step in range(x.size(0)):
            out = self.net(x[step])
            spk_rec.append(out)
        return torch.stack(spk_rec)

def calculate_iou(pred_box, target_box):
    xA = max(pred_box[0], target_box[0])
    yA = max(pred_box[1], target_box[1])
    xB = min(pred_box[2], target_box[2])
    yB = min(pred_box[3], target_box[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(1, (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]))
    boxBArea = max(1, (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

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

def train_model(trainloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    spike_grad = surrogate.sigmoid()
    beta = 0.5

    net = nn.Sequential(
        SCNN(beta, spike_grad),
        FIL(14336, 4096, 4, beta=beta, spike_grad=spike_grad)
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()

    num_epochs = 20

    loss_hist = []
    iou_hist = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_iou = 0
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.squeeze(1)
            targets = targets.to(device)

            net.train()
            spk_rec = net(data)
            sample_out = spk_rec[-1]
            sample_target = targets[-1]

            optimizer.zero_grad()
            loss = loss_fn(spk_rec, targets)
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            epoch_loss += loss.item()

            pred_box = sample_out.detach().cpu().numpy().flatten()
            actual_box = sample_target.detach().cpu().numpy().flatten()
            iou = calculate_iou(pred_box, actual_box)
            iou_hist.append(iou)
            epoch_iou += iou

            if i < 5 and epoch == 0:
                image = data[:, 0].cpu()
                image = image[-1]
                visualize_bounding_boxes([image], [pred_box], [actual_box])

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss.item():.2f}")
            print("Predicted Bounding Box: ", pred_box)
            print("Actual Bounding Box: ", actual_box)
            print(f"IoU: {iou * 100:.2f}%")
            print()

        avg_loss = epoch_loss / len(trainloader)
        avg_iou = epoch_iou / len(trainloader)
        loss_hist.append(avg_loss)
        iou_hist.append(avg_iou)

    epochs = list(range(1, num_epochs + 1))

    avg_loss_per_epoch = [np.mean(loss_hist[i * len(trainloader):(i + 1) * len(trainloader)]) for i in range(num_epochs)]
    avg_iou_per_epoch = [np.mean(iou_hist[i * len(trainloader):(i + 1) * len(trainloader)]) for i in range(num_epochs)]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_loss_per_epoch, label="Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_iou_per_epoch, label="Average IoU per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.tight_layout()
    plt.show()

def forward_pass(net, data):
    return net(data)