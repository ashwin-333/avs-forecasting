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

def calculate_coverage(pred_box, target_box):
    xA = max(pred_box[0], target_box[0])
    yA = max(pred_box[1], target_box[1])
    xB = min(pred_box[2], target_box[2])
    yB = min(pred_box[3], target_box[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    actualArea = max(1, (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]))
    
    coverage = (interArea / actualArea) * 100
    return coverage

def train_model(trainloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # neuron and simulation parameters
    spike_grad = surrogate.sigmoid()
    beta = 0.5

    # Initialize Network
    net = nn.Sequential(nn.Conv2d(1, 64, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(4),
                        nn.Conv2d(64, 128, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(4),
                        nn.Flatten(),
                        nn.Linear(56448, 4),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()

    num_epochs = 3
    num_steps = 346
    # num_iters = 50

    loss_hist = []
    train_acc_hist = []

    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)
            targets = targets.squeeze(1)  # becomes batch x box (only works when theres one bounding box)

            net.train()
            spk_rec = forward_pass(net, data, num_steps)
            
            total_spks = torch.sum(spk_rec, 0)  # now its batch x outputs (spks * num_steps)
            #idea is to sum up spikes and compare to targets.

            #Gradient calculation + weight update
            #assumes batch is 1
            if total_spks.size(0) != 1 or targets.size(0) != 1:
                raise ValueError("Batch size is not 1")
            
            optimizer.zero_grad()
            loss = loss_fn(total_spks, targets)
            loss.backward()
            optimizer.step()
      
            # Store loss history for future plotting
            loss_hist.append(loss.item())

            # Calculate IoU and Coverage
            pred_box = total_spks.detach().cpu().numpy().flatten()
            actual_box = targets.detach().cpu().numpy().flatten()
            iou = calculate_iou(pred_box, actual_box)
            coverage = calculate_coverage(pred_box, actual_box)

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss.item():.2f}")
            print("output spikes (bounding box): ", total_spks)
            print("target: ", targets)
            print("Predicted Bounding Box: ", pred_box)
            print("Actual Bounding Box: ", actual_box)
            print(f"IoU: {iou * 100:.2f}%")
            print(f"Coverage: {coverage:.2f}%\n")

def forward_pass(net, data, num_steps):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net
    
    #this is the time step. Same input is passed in.
    for step in range(num_steps):
        spk_out, _ = net(data)
        spk_rec.append(spk_out)
    
    #print(spk_rec)
    return torch.stack(spk_rec)  # dim(steps, batch, outputs)
