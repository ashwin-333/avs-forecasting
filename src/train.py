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



def train_model(trainloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # neuron and simulation parameters
    spike_grad = surrogate.sigmoid()
    beta = 0.5

    # Initialize Network
    net = nn.Sequential(nn.Conv2d(1, 64, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(3, stride = 2),
                        nn.Conv2d(64, 128, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(3, stride = 2),
                        nn.Conv2d(128, 256, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(3, stride = 2),
                        nn.Conv2d(256, 512, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(3, stride = 2),
                        nn.Conv2d(512, 512, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(3, stride = 2),
                        nn.Flatten(),
                        nn.Linear(17920, 4096),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Linear(4096, 4)
                        ).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()

    num_epochs = 3
    num_steps = 5

    loss_hist = []
    train_acc_hist = []

    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)

            targets = targets.squeeze(1)  # becomes batch x box (only works when theres one bounding box)
            targets = targets.to(device)

            net.train()
            out_arr = forward_pass(net, data, num_steps)
            
            out_steps = out_arr.sum(0) #sums up outputs from all steps. Divide by steps will give us mean and what our predicted will be
            targets_steps = targets.mul(num_steps)


            #assumes batch is 1
            if targets_steps.size(0) != 1 or out_steps.size(0) != 1:
                raise ValueError("Batch size is not 1")
            
            optimizer.zero_grad()
            loss = loss_fn(out_steps, targets_steps) / num_steps #check this
            loss.backward()
            optimizer.step()
      

            loss_hist.append(loss.item())

            out_avg = out_steps.div(num_steps)

            # Calculate IoU and Coverage
            pred_box = out_avg.detach().cpu().numpy().flatten()
            actual_box = targets.detach().cpu().numpy().flatten()
            iou = calculate_iou(pred_box, actual_box)

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss.item():.2f}")
            print("Predicted Bounding Box: ", pred_box)
            print("Actual Bounding Box: ", actual_box)
            print(f"IoU: {iou * 100:.2f}%")
            print()

def forward_pass(net, data, num_steps):
    out_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net
    
    #this is the time step. Same input is passed in.
    for step in range(num_steps):
        out = net(data)
        out_rec.append(out)

    return torch.stack(out_rec)  # dim(steps, batch, outputs)

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