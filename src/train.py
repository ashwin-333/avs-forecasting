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

def train_model(trainloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    spike_grad = surrogate.sigmoid()
    beta = 0.5

    net = nn.Sequential(nn.Conv2d(2, 64, 3),
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
                        nn.Flatten(),
                        nn.Linear(14336, 4096),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Linear(4096, 4),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism="none"),
                        ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()

    num_epochs = 3

    loss_hist = []
    iou_hist = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_iou = 0
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device) # (T x B x C x H x W)
            targets = targets.squeeze(1) 
            targets = targets.to(device) # B x 4

            net.train()
            spk_rec = forward_pass(net, data)

            optimizer.zero_grad()
            loss = loss_fn(spk_rec, targets)
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())
            epoch_loss += loss.item()

            pred_box = spk_rec.detach().cpu().numpy().flatten()
            actual_box = targets.detach().cpu().numpy().flatten()
            iou = calculate_iou(pred_box, actual_box)
            iou_hist.append(iou)
            epoch_iou += iou

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
    spk_rec = []
    utils.reset(net)
    for step in range(data.size(0)):
        _, mem_rec = net(data[step])
        print(mem_rec.shape)
        spk_rec.append(mem_rec)
    print("hi", torch.sum(torch.stack(spk_rec), dim=0))
    return torch.sum(torch.stack(spk_rec), dim=0)

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
